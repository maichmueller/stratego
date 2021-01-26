from typing import Optional, Union

from stratego.learning import ReplayContainer, AlphaZeroMemory
from stratego.algorithms.mcts import MCTS

from stratego import Game
import stratego.arena as arena
from stratego.engine import Logic, Team, ActionMap, Action, Status
from stratego.agent import RLAgent

from copy import deepcopy

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import numpy as np

from pickle import Pickler, Unpickler
import os
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Lock

from stratego.utils import slice_kwargs, RollingMeter


class Teacher:
    def __init__(
        self,
        student: RLAgent,
        action_map: ActionMap,
        logic: Logic = Logic(),
        num_iterations: int = 100,
        model_folder: str = "./checkpoints/models",
        train_data_folder: str = "./checkpoints/data",
        **kwargs,
    ):

        self.model_folder = model_folder
        self.train_data_folder = train_data_folder
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(train_data_folder):
            os.makedirs(train_data_folder)

        self.n_iters = num_iterations

        assert isinstance(
            student, RLAgent
        ), f"Student agent to coach has to be of type '{RLAgent}'. Given type '{type(self.student).__name__}'"
        self.student: RLAgent = student
        self.student_mirror: RLAgent = deepcopy(student)  # a copy of the student to fight against
        self.logic = logic
        self.action_map = action_map
        self.game = Game(self.student, self.student_mirror, logic=logic, **kwargs)

    def teach(self, *args, **kwargs):
        """
        Teach the reinforcement learning agent according to the strategy defined in the Teacher child.
        """
        raise NotImplementedError

    def load_train_data(self, filename: str):
        data_filepath = os.path.join(self.train_data_folder, filename, ".data")
        with open(data_filepath, "rb") as f:
            train_data = Unpickler(f).load()
        return train_data

    def save_train_data(self, data, filename: str):
        folder = self.train_data_folder
        filename = os.path.join(folder, filename, ".data")
        with open(filename, "wb+") as f:
            Pickler(f).dump(data)


class AZTeacher(Teacher):
    def __init__(
        self,
        student: RLAgent,
        action_map: ActionMap,
        logic: Logic = Logic(),
        num_iterations: int = 100,
        num_selfplay_episodes: int = 100,
        acceptance_rate: float = 0.55,
        mcts_simulations: int = 100,
        temperature: int = 100,
        model_folder: str = "./checkpoints/models",
        train_data_folder: str = "./checkpoints/data",
        seed: Optional[Union[int, np.random.Generator]] = None,
        **kwargs,
    ):
        super().__init__(
            student,
            action_map,
            logic,
            num_iterations,
            model_folder,
            train_data_folder,
            **kwargs,
        )
        self.n_iters = num_iterations
        self.n_episodes = num_selfplay_episodes
        self.n_mcts_sim = mcts_simulations

        self.acceptance_rate = acceptance_rate

        self.model_folder = model_folder
        self.train_data_folder = train_data_folder
        self.temp_thresh = temperature

        self.skip_first_self_play = False
        self.rng = np.random.default_rng(seed)

    def selfplay_episode(
        self, mcts, memory_capacity: int = int(1e5), lock: Optional[Lock] = None
    ):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to a list.
        The game is played until the game ends. After the game ends, the outcome
        of the game is used to assign the correct values to each entry in the
        selfplay data.
        It sets temperature = 1 if episode < temp_threshold, and thereafter
        uses temperature=0.

        Parameters
        ----------
        mcts: MCTS,
            the Monte-Carlo Tree Search object.
        memory_capacity: int,
            the capacity of the replay
        lock: Lock,
            the lock to use when multiprocessing is in play.

        Returns
        -------
        selfplay_data: ReplayMemory,
            a list of examples of the form (board,policy,v) policy is the MCTS informed
            policy vector, v is +1 if the player eventually won the game, else -1.
        """

        ep_step = 0
        replays = ReplayContainer(memory_capacity, AlphaZeroMemory)

        state = self._new_state_safe(lock)

        while True:
            ep_step += 1
            policy = mcts.policy(
                state, agent=self.student, temperature=int(ep_step < self.temp_thresh)
            )
            action = self.action_map[self.rng.choice(len(policy), p=policy)]
            if self.student.team == Team.red:
                # invert the action's effect to get the mirrored action, since the chosen
                # action is always chosen from the perspective of team Blue.
                action = Action(action.actor, -action.effect)
            move = self.action_map.action_to_move(action, state, self.student.team)

            replays.push(deepcopy(state), policy, None, self.student.team)

            self.logic.execute_move(state, move=move)
            status = self.logic.check_terminal(state, self.game.specs)

            if status != Status.ongoing:
                for entry in replays:
                    if self.student.team == state.active_team:
                        # logic.execute_move is expected to increase the turn counter after
                        # executing the move and thus change the active team afterwards.
                        # Therefore the team that provided the move is actually the one,
                        # which is not currently active.
                        perspective = -1
                    else:
                        perspective = 1
                    entry.value = status.value * perspective
                return replays

    def _new_state_safe(self, lock: Optional[Lock] = None):
        if lock is not None:
            lock.lock()
        self.game.reset()
        state = deepcopy(self.game.state)
        if lock is not None:
            lock.unlock()
        return state

    def teach(
        self,
        memory_capacity: int = 100000,
        load_checkpoint_data: bool = False,
        load_checkpoint_model: bool = False,
        multiprocess: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Performs `num_iters` many iterations with `n_episodes` episodes of self-play in each
        iteration. After every iteration, it retrains the neural network with the data in
        selfplay_data (which has a maximum length of `memory_capacity`).
        It then pits the new neural network against the old one and accepts it
        only if it wins with a rate greater than the threshold.
        """
        network = self.student.network
        checkpoint_found = False
        skip_initial_selfplay = load_checkpoint_data and not load_checkpoint_model

        if load_checkpoint_data:
            i = 0
            while True:
                if os.path.isfile(
                    self.train_data_folder + f"checkpoint_{i}.pth.tar" + ".data"
                ):
                    checkpoint = f"checkpoint_{i}.pth.tar"
                    checkpoint_found = True
                    i += 1
                else:
                    break
        if load_checkpoint_model and checkpoint_found:
            self.load_train_data(checkpoint)
            if os.path.isfile(self.model_folder + f"best.pth.tar"):
                network.load_checkpoint(self.model_folder, f"best.pth.tar")

        selfplay_data = ReplayContainer(memory_capacity, AlphaZeroMemory)
        selfplay_data_tensors = ReplayContainer(memory_capacity, AlphaZeroMemory)
        action_map = ActionMap(self.game.specs)
        mcts_kwargs = slice_kwargs(MCTS.__init__, kwargs)
        arena_kwargs = slice_kwargs(arena.fight, kwargs)

        for i in range(self.n_iters):

            print("\n------ITER " + str(i) + "------", flush=True)

            if not skip_initial_selfplay or i > 1:
                network.network.share_memory()

                if multiprocess:
                    pbar = tqdm(total=self.n_episodes)
                    pbar.set_description("Selfplay Episode")
                    lock = Lock()
                    with ProcessPoolExecutor(
                        max_workers=kwargs.pop("cpu_count", cpu_count())
                    ) as executor:
                        futures = list(
                            (
                                executor.submit(
                                    self.selfplay_episode,
                                    mcts=MCTS(
                                        network,
                                        action_map=action_map,
                                        logic=self.logic,
                                        n_mcts_sims=self.n_mcts_sim,
                                        **mcts_kwargs,
                                    ),
                                    lock=lock,
                                    memory_capacity=memory_capacity,
                                )
                                for i in range(self.n_episodes)
                            )
                        )
                        for future in as_completed(futures):
                            pbar.update(1)
                            results = future.result()
                            selfplay_data.extend(*results)
                            for result in results:
                                result.state = self.student.state_to_tensor(
                                    result.state, perspective=self.student.team
                                )
                else:
                    for _ in tqdm(range(self.n_episodes)):
                        # bookkeeping + plot progress through tqdm
                        # reset search tree
                        mcts = MCTS(
                            network,
                            action_map=action_map,
                            logic=self.logic,
                            n_mcts_sims=self.n_mcts_sim,
                            **mcts_kwargs,
                        )
                        selfplay_data.extend(
                            self.selfplay_episode(mcts, memory_capacity=memory_capacity)
                        )

            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            if not skip_initial_selfplay or i > 0:
                self.save_train_data(i - 1, filename=kwargs.pop("data_filename", f"iteration_{i}"))

            # training new network, keeping a copy of the old one
            network.save_checkpoint(folder=self.model_folder, filename="temp.pth.tar")
            self.student_mirror.network.load_checkpoint(
                folder=self.model_folder, filename="temp.pth.tar"
            )

            self.train(selfplay_data, 100, batch_size=4096)

            print("\nPITTING AGAINST PREVIOUS VERSION")

            ag_0_wins, ag_1_wins, draws = arena.fight(
                game_env=self.game, n_fights=self.n_iters, **arena_kwargs
            )

            print(
                f"Wins / losses of new model: {ag_0_wins} / {ag_1_wins} "
                f"({100 * ag_0_wins / (ag_1_wins + ag_0_wins):.1f}%) | draws: {draws}"
            )
            if (
                ag_0_wins + ag_1_wins > 0
                and float(ag_0_wins) / (ag_0_wins + ag_1_wins) < self.acceptance_rate
            ):
                print("REJECTING NEW MODEL\n")
                network.load_checkpoint(
                    folder=self.model_folder, filename="temp.pth.tar"
                )
            else:
                print("ACCEPTING NEW MODEL\n")
                network.save_checkpoint(
                    folder=self.model_folder, filename=f"checkpoint_{i}.pth.tar"
                )

    def train(self, replays: ReplayContainer, epochs, batch_size=128):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        network = self.student.network
        optimizer = optim.Adam(network.parameters())
        for _ in tqdm(range(epochs), desc="Training epoch"):
            network.train()
            pi_losses = RollingMeter()
            v_losses = RollingMeter()

            data_loader = DataLoader(TensorDataset(replays), batch_size=batch_size)
            batch_bar = tqdm(data_loader)
            for batch_idx, batch in enumerate(data_loader):
                boards, pis, vs, _ = batch

                boards = torch.cat(boards).to(device=self.device)
                target_pis = torch.Tensor(np.array(pis), device=self.device)
                target_vs = torch.Tensor(
                    np.array(vs).astype(np.float64), device=self.device
                )
                # compute output
                out_pi, out_v = network(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.push(l_pi.item(), boards.size(0))
                v_losses.push(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # plot progress
                batch_bar.set_description(
                    "Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}".format(
                        lpi=pi_losses.avg,
                        lv=v_losses.avg,
                    )
                )

    @classmethod
    def loss_pi(cls, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    @classmethod
    def loss_v(cls, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


if __name__ == "__main__":
    c = AZTeacher(
        agent.AlphaZero(0),
        num_selfplay_episodes=1000,
        mcts_simulations=100,
        game_size="small",
    )
    c.teach(
        load_checkpoint_data=True,
        load_checkpoint_model=True,
        skip_first_self_play=False,
        multiprocess=False,
    )

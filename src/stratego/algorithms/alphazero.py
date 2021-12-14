from typing import Optional, Union

from .algorithms import DeepLearningAlgorithm

from stratego.learning import (
    Experience,
    AlphaZeroMemory
)
from stratego.algorithms.mcts import MCTS

from stratego.game import Game
import stratego.arena as arena
from stratego.core import Team, ActionMap, Action, Status
from stratego.agent import AlphaZeroAgent

from copy import deepcopy

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import numpy as np

import os
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Lock

import stratego.utils as utils


class AlphaZeroAlgorithm(DeepLearningAlgorithm):
    def __init__(
        self,
        game: Game,
        student: AlphaZeroAgent,
        action_map: ActionMap,
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
            game,
            student,
            action_map,
            model_folder,
            train_data_folder,
            **kwargs,
        )

        assert isinstance(
            game.agents[self.student.team], AlphaZeroAgent
        ), "Student needs to be an AlphaZero Agent."

        self.student_mirror = deepcopy(game.agents[self.student.team.opponent()])

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
        replays = Experience(memory_capacity, AlphaZeroMemory)

        state = self._new_state(lock)

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

            self.game.logic.execute_move(state, move=move)
            status = self.game.logic.check_terminal(state, self.game.specs)

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

    def run(
        self,
        memory_capacity: int = 100000,
        load_checkpoint_data: bool = False,
        load_checkpoint_model: bool = False,
        batch_size: int = 4096,
        n_epochs: int = 100,
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
        model = self.student.model
        skip_initial_selfplay = load_checkpoint_data and not load_checkpoint_model

        if load_checkpoint_data:
            checkpoint_fname = kwargs.pop("checkpoint_fname", "checkpoint")
            checkpoint = None
            if not os.path.isfile(
                os.path.join(
                    self.train_data_folder, f"{checkpoint_fname}_0.pth.tar" + ".data"
                )
            ):
                raise ValueError(
                    f"No checkpoint file found with name: {checkpoint_fname}"
                )
            i = 0
            while True:
                if os.path.isfile(
                    os.path.join(
                        self.train_data_folder,
                        f"{checkpoint_fname}_{i}.pth.tar" + ".data",
                    )
                ):
                    checkpoint = f"{checkpoint_fname}_{i}.pth.tar"
                    i += 1
                else:
                    # we have found the last iteration data package, we can therefore stop searching
                    # and continue from here
                    break

            self.load_train_data(checkpoint)
            model_fpath = os.path.join(
                self.model_folder, f"{kwargs.pop('model_fname', 'best')}.pth.tar"
            )
            if os.path.isfile(model_fpath):
                model.load_checkpoint(model_fpath)

        selfplay_data = Experience(memory_capacity, AlphaZeroMemory)
        selfplay_data_tensors = Experience(memory_capacity, AlphaZeroMemory)
        mcts_kwargs = utils.slice_kwargs(MCTS.__init__, kwargs)
        arena_kwargs = utils.slice_kwargs(arena.fight, kwargs)

        for i in range(self.n_iters):

            print("\n------ITER " + str(i) + "------", flush=True)

            if not skip_initial_selfplay or i > 1:
                model.network.share_memory()
                self.generate_selfplay_data(
                    model,
                    selfplay_data,
                    selfplay_data_tensors,
                    multiprocess,
                    memory_capacity,
                    kwargs.pop("cpu_count", cpu_count()),
                    mcts_kwargs,
                )

            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            if not skip_initial_selfplay or i > 0:
                self.save_train_data(
                    i - 1, filename=kwargs.pop("data_filename", f"iteration_{i}")
                )

            # training new network, keeping a copy of the old one
            model.save_checkpoint(folder=self.model_folder, filename="temp.pth.tar")
            self.student_mirror.model.load_checkpoint(
                folder=self.model_folder, filename="temp.pth.tar"
            )

            self.train(selfplay_data, n_epochs, batch_size=batch_size, device=device)

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
                model.load_checkpoint(folder=self.model_folder, filename="temp.pth.tar")
            else:
                print("ACCEPTING NEW MODEL\n")
                model.save_checkpoint(
                    folder=self.model_folder, filename=f"checkpoint_{i}.pth.tar"
                )

    def generate_selfplay_data(
        self,
        model: torch.nn.Module,
        selfplay_container: Experience,
        selfplay_tensors_container: Experience,
        multiprocess: bool,
        memory_capacity: int,
        n_cpus: int,
        mcts_kwargs,
    ):
        def append(outcomes):
            nonlocal self
            selfplay_container.extend(*outcomes)
            for outcome in outcomes:
                selfplay_tensors_container.push(
                    self.student.state_to_tensor(
                        outcome.state, perspective=self.student.team
                    ),
                    *outcome[1:],
                )

        if multiprocess:
            pbar = tqdm(total=self.n_episodes)
            pbar.set_description("Selfplay Episode")
            lock = Lock()
            with ProcessPoolExecutor(max_workers=n_cpus) as executor:
                futures = list(
                    (
                        executor.submit(
                            self.selfplay_episode,
                            mcts=MCTS(
                                model,
                                action_map=self.action_map,
                                logic=self.game.logic,
                                n_mcts_sims=self.n_mcts_sim,
                                **mcts_kwargs,
                            ),
                            lock=lock,
                            memory_capacity=memory_capacity,
                        )
                        for _ in range(self.n_episodes)
                    )
                )
                for future in as_completed(futures):
                    pbar.update(1)
                    append(future.result())

        else:
            for _ in tqdm(range(self.n_episodes), "Selfplay Episode"):
                # new search tree
                mcts = MCTS(
                    model,
                    action_map=self.action_map,
                    logic=self.game.logic,
                    n_mcts_sims=self.n_mcts_sim,
                    **mcts_kwargs,
                )
                append(self.selfplay_episode(mcts, memory_capacity=memory_capacity))

    def train(
        self, replays: Experience, epochs: int, batch_size: int, device: str
    ):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        model = self.student.model
        optimizer = optim.Adam(model.parameters())
        for _ in tqdm(range(epochs), desc="Training epoch"):
            model.run()
            pi_losses = utils.RollingMeter()
            v_losses = utils.RollingMeter()

            data_loader = DataLoader(
                TensorDataset(replays.memory), batch_size=batch_size
            )
            batch_bar = tqdm(data_loader)
            for batch_idx, batch in enumerate(data_loader):
                boards, pis, vs, _ = batch

                boards = torch.cat(boards).to(device)
                target_pis = torch.tensor(np.array(pis), device=device)
                target_vs = torch.tensor(np.array(vs).astype(np.float64), device=device)
                # compute output
                out_pi, out_v = model(boards)
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

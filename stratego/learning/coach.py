from stratego.learning import ReplayContainer, AlphaZeroMemory
from .mcts import MCTS

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
from collections import namedtuple
import os
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


class Coach:
    def __init__(
        self,
        game_env: Game,
        team_to_train: Team,
        action_map: ActionMap,
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

        self.game = game_env
        student = game_env.agents[team_to_train]
        assert isinstance(
            student, RLAgent
        ), f"Student agent to coach has to be of type '{RLAgent}'. Given type '{type(self.student).__name__}'"
        self.student: RLAgent = student
        self.action_map = action_map

    def teach(self, *args, **kwargs):
        """
        Teach the reinforcement learning agent according to the strategy defined in the sub-coach.
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


class AZCoach(Coach):
    def __init__(
        self,
        game_env: Game,
        team_to_train: Team,
        action_map: ActionMap,
        num_iterations: int = 100,
        num_selfplay_episodes: int = 100,
        acceptance_rate: float = 0.55,
        mcts_simulations: int = 100,
        temperature: int = 100,
        model_folder: str = "./checkpoints/models",
        train_data_folder: str = "./checkpoints/data",
        **kwargs,
    ):
        super().__init__(
            game_env,
            team_to_train,
            action_map,
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

    def selfplay_episode(
        self, mcts, state=None, replay_capacity: int = int(1e5), logic: Logic = Logic()
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
        state: State,
            the state on which we are going to search.
        replay_capacity: int,
            the capacity of the replay
        logic: Logic,
            the logic element handling state manipulations according to the chosen ruleset.

        Returns
        -------
        selfplay_data: ReplayMemory,
            a list of examples of the form (board,policy,v) policy is the MCTS informed
            policy vector, v is +1 if the player eventually won the game, else -1.
        """

        self.game.reset()

        if state is None:
            state = deepcopy(self.game.state)

        ep_step = 0
        replays = ReplayContainer(replay_capacity, AlphaZeroMemory)

        while True:
            ep_step += 1
            policy = mcts.policy(
                state, agent=self.student, temperature=int(ep_step < self.temp_thresh)
            )
            action = self.action_map[np.random.choice(len(policy), p=policy)]
            if self.student.team == Team.red:
                # invert the action's effect to get the mirrored action, since the chosen
                # action is always chosen from the perspective of team Blue.
                action = Action(action.actor, -action.effect)
            move = self.action_map.action_to_move(action, state, self.student.team)

            replays.push(deepcopy(state), policy, None, self.student.team)

            logic.execute_move(state, move=move)
            status = logic.check_terminal(state)

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
                return

    def teach(
        self,
        memory_capacity: int = 100000,
        load_checkpoint_data: bool = False,
        load_checkpoint_model: bool = False,
        multiprocess: bool = False,
        device: str = "cpu",
    ):
        """
        Performs num_iters many iterations with num_episodes episodes of self-play in each
        iteration. After every iteration, it retrains the neural network with
        examples in selfplay_data (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        network = self.student.model
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

        for i in range(self.n_iters):

            print("\n------ITER " + str(i) + "------", flush=True)

            if not self.skip_first_self_play or i > 1:
                network.to_device()
                network.net.share_memory()

                if multiprocess:
                    pbar = tqdm(total=self.n_episodes)
                    pbar.set_description("Creating self-play training turns")
                    with ProcessPoolExecutor(max_workers=cpu_count() // 2) as executor:
                        futures = list(
                            (
                                executor.submit(
                                    self.self_play_episode,
                                    mcts=MCTS(network, n_mcts_sims=self.n_mcts_sim),
                                    reset_game=False,
                                    state=deepcopy(self.game.reset().state),
                                )
                                for i in range(self.n_episodes)
                            )
                        )
                        for future in as_completed(futures):
                            pbar.update(1)
                            selfplay_data.extend(future.result())
                else:
                    for _ in tqdm(range(self.n_episodes)):
                        # bookkeeping + plot progress through tqdm
                        # reset search tree
                        mcts = MCTS(network, n_mcts_sims=self.n_mcts_sim)
                        selfplay_data.extend(
                            self.self_play_episode(mcts, reset_game=True)
                        )

            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            if not self.skip_first_self_play or i > 0:
                self.save_train_data(i - 1)
            # convert the bard to a state rep
            train_exs = []
            for tr_turn in selfplay_data:
                train_exs.push(
                    self.game.agents[0].state_to_tensor(tr_turn.board),
                    tr_turn.pi,
                    tr_turn.v,
                    tr_turn.player,
                )
            self.train_examples.extend(train_exs)

            # training new network, keeping a copy of the old one
            network.save_checkpoint(folder=self.model_folder, filename="temp.pth.tar")
            self.opp_net.load_checkpoint(
                folder=self.model_folder, filename="temp.pth.tar"
            )

            network.train(self.train_examples, 100, batch_size=4096)

            print("\nPITTING AGAINST PREVIOUS VERSION")

            ag_0_wins, ag_1_wins, draws = arena.fight(
                game_env=self.game, n_fights=self.n_iters, **kwargs
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

    def train(self, examples: ReplayContainer, epochs, batch_size=128):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        network = self.student.model
        optimizer = optim.Adam(network.parameters())
        for _ in tqdm(range(epochs), desc="Training epoch"):
            network.train()
            pi_losses = RollingMeter()
            v_losses = RollingMeter()

            data_loader = DataLoader(TensorDataset(examples), batch_size=batch_size)
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
    c = AZCoach(
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

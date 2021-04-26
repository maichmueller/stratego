from abc import abstractmethod, ABC
from typing import Optional, Union

from stratego.learning import (
    ReplayContainer,
    AlphaZeroMemory,
    DQNMemory,
    RandomActionScheduler,
)
from stratego.algorithms.mcts import MCTS

from stratego import Game
import stratego.arena as arena
from stratego.engine import Logic, Team, ActionMap, Action, Status
from stratego.agent import RLAgent, AZAgent, DQNAgent

from copy import deepcopy

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.functional import F
import numpy as np

from pickle import Pickler, Unpickler
import os
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Lock

import stratego.utils as utils


class Teacher(ABC):
    def __init__(
        self,
        student: RLAgent,
        action_map: ActionMap,
        logic: Logic = Logic(),
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

        assert isinstance(
            student, RLAgent
        ), f"Student agent to coach has to be of type '{RLAgent}'. Given type '{type(self.student).__name__}'"
        self.student: RLAgent = student
        self.student_mirror: RLAgent = deepcopy(
            student
        )  # a copy of the student to fight against
        self.action_map = action_map
        self.game = Game(self.student, self.student_mirror, logic=logic, **kwargs)

    @abstractmethod
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

    def _new_state_threadsafe(self, lock: Optional[Lock] = None):
        """
        Resets the game and returns a deepcopy of the new state in a thread-safe manner, if the lock is provided.
        """
        gen_state = lambda: deepcopy(self.game.reset().state)
        if lock is not None:
            lock.lock()
            state = gen_state()
            lock.unlock()
        else:
            state = gen_state()
        return state


class AZTeacher(Teacher):
    def __init__(
        self,
        student: AZAgent,
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

        state = self._new_state_threadsafe(lock)

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

    def teach(
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
                model.load_checkpoint(self.model_folder, f"best.pth.tar")

        selfplay_data = ReplayContainer(memory_capacity, AlphaZeroMemory)
        selfplay_data_tensors = ReplayContainer(memory_capacity, AlphaZeroMemory)
        action_map = ActionMap(self.game.specs)
        mcts_kwargs = utils.slice_kwargs(MCTS.__init__, kwargs)
        arena_kwargs = utils.slice_kwargs(arena.fight, kwargs)

        for i in range(self.n_iters):

            print("\n------ITER " + str(i) + "------", flush=True)

            if not skip_initial_selfplay or i > 1:
                model.network.share_memory()

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
                                        model,
                                        action_map=action_map,
                                        logic=self.game.logic,
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
                                selfplay_data_tensors.push(
                                    self.student.state_to_tensor(
                                        result.state, perspective=self.student.team
                                    ),
                                    *result[1:],
                                )

                else:
                    for _ in tqdm(range(self.n_episodes)):
                        # reset search tree
                        mcts = MCTS(
                            model,
                            action_map=action_map,
                            logic=self.game.logic,
                            n_mcts_sims=self.n_mcts_sim,
                            **mcts_kwargs,
                        )
                        new_replays = self.selfplay_episode(
                            mcts, memory_capacity=memory_capacity
                        )
                        selfplay_data.extend(new_replays)
                        for result in new_replays:
                            selfplay_data_tensors.push(
                                self.student.state_to_tensor(
                                    result.state, perspective=self.student.team
                                ),
                                *result[1:],
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

    def train(
        self, replays: ReplayContainer, epochs: int, batch_size: int, device: str
    ):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        model = self.student.model
        optimizer = optim.Adam(model.parameters())
        for _ in tqdm(range(epochs), desc="Training epoch"):
            model.train()
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


class DQNTeacher(Teacher):
    """
    A Deep Q-Network Teacher using the Double Q Learning strategy[1] and Dueling Networks[2].

    References
    ----------
    [1] Van Hasselt, Hado, Arthur Guez, and David Silver.
        "Deep reinforcement learning with double q-learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 30. No. 1. 2016.
        https://ojs.aaai.org/index.php/AAAI/article/download/10295/10154
    [2] Wang, Ziyu, et al.
        "Dueling network architectures for deep reinforcement learning."
        International conference on machine learning. PMLR, 2016.
        http://proceedings.mlr.press/v48/wangf16.pdf
    [3] Fujimoto Herke, van Hoof, David Meger
        "Addressing Function Approximation Error in Actor-Critic Methods"
        https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(
        self,
        epsilon_scheduler: RandomActionScheduler,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.epsilon_scheduler = epsilon_scheduler

    def train(
        self,
        n_epochs: int,
        batch_size: int = 4096,
        gamma: float = 0.99,
        memory_capacity: int = 50000,
        device: str = "cpu",
        seed: Union[int, np.random.Generator] = None,
    ):
        """
        Trains a reinforcement agent, acting according to the agents model
        or randomly (with exponentially decaying probability p_random)
        Each transition (state, action, next_state, reward) is stored in a memory.
        Each step in the environment is followed by a learning phase:
            a batch of memories is used to optimize the network model

        Parameters
        ----------
        n_epochs
        batch_size
        memory_capacity
        device
        seed

        Returns
        -------

        """
        if not isinstance(self.student, DQNAgent):
            raise ValueError(
                f"Provided student is not a DQN agent. Given: {type(self.student).__name__}"
            )
        rng = utils.rng_from_seed(seed)
        replays = ReplayContainer(memory_capacity, DQNMemory, rng)
        for ep in range(n_epochs):
            self.game.reset()
            state = self.game.state
            state_tensor = self.student.state_to_tensor(state)
            while True:
                p_random = self.epsilon_scheduler(ep)
                do_random = bool(rng.choice(2, p=[1 - p_random, p_random]))
                # random action chosen with probability p_random
                if do_random:
                    action = self.student.select_random_action(state, self.game.logic)
                else:
                    policy = self.student.model(state_tensor)
                    action = self.student.select_action(policy, deterministic=True)
                move = self.action_map.action_to_move(action, state, self.student.team)

                # environment step for action
                status = self.game.run_step(move)
                reward = torch.tensor(
                    self.student.reward, dtype=torch.float, device=device
                )
                self.student.reward = 0

                # save transition as memory and optimize model
                if status != Status.ongoing:  # if terminal state
                    next_state = None
                else:
                    next_state = self.student.state_to_tensor(
                        state, perspective=self.student.team
                    )

                replays.push(
                    state, action, next_state, reward
                )  # store the transition in memory
                state = next_state  # move to the next state
                # one step of optimization of target network
                self._optimize_model(
                    self.student.model,
                    replays.sample(batch_size),
                    gamma,
                    device,
                )

    def _optimize_model(self, model: torch.nn.Module, batch: np.ndarray, gamma: float, device: str):
        """
        Sample batch from memory of environment transitions and train network to fit the
        temporal difference TD(0) Q-value approximation
        """
        model.train()

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = []
        non_final_idx = []
        non_final_next_states = []

        batch_size = len(batch)

        for idx, state in enumerate(batch.next_state):
            if state is not None:
                non_final_mask.append(True)
                non_final_idx.append(idx)
                non_final_next_states.append(state)
            else:
                non_final_mask.append(False)
        non_final_mask = torch.ByteTensor(non_final_mask)
        # non_final_idx = np.array(non_final_idx)
        non_final_next_states = torch.cat(non_final_next_states)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = (
            torch.zeros(batch_size).float().to(device)
        )  # zero for terminal states

        # what would the model predict
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
        with torch.no_grad():
            expected_state_action_values = (
                next_state_values * gamma
            ) + reward_batch  # compute the expected Q values

        loss = F.smooth_l1_loss(
            state_action_values.view(-1), expected_state_action_values.view(-1)
        )  # compute Huber loss

        # optimize network
        optimizer.zero_grad()  # optimize towards expected q-values
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

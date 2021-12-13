from enum import Enum
from typing import Optional, Union, Type, Dict, Any, Callable, List

from .algorithms import Algorithm
from stratego.learning import (
    Experience,
    DQNMemory,
    ExplorationScheduler,
    PolicyMode,
)

from stratego.core import Status
from stratego.agent import DQNAgent

from copy import deepcopy

import torch
from torch.functional import F
import numpy as np

import stratego.utils as utils
from .algorithms import DeepLearningAlgorithm
from stratego.agent import RLAgent, DRLAgent
from stratego.core import ActionMap
from stratego.game import Game


class DQNAlgorithm(Algorithm):
    """
    A Deep Q-Network Teacher using the Double Q Learning strategy[1] if activated.
    This can also be used to test a dueling network architecture[2].

    References
    ----------
    [1] Van Hasselt, Hado, Arthur Guez, and David Silver.
        "Deep reinforcement learning with double q-learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 30. No. 1. 2016.
        https://ojs.aaai.org/index.php/AAAI/article/download/10295/10154
    """

    def __init__(
            self,
            game: Game,
            student: DQNAgent,
            action_map: ActionMap,
            epsilon_scheduler: ExplorationScheduler,
            double_q: bool = True,
            model_folder: str = "./checkpoints/models",
            train_data_folder: str = "./checkpoints/data",
    ):
        super().__init__(
            game, student, action_map
        )
        self.model_folder = model_folder
        self.train_data_folder = train_data_folder
        assert isinstance(
            self.student, DQNAgent
        ), "Student agent to teach is not a DQN agent."
        # reassigned only to silence warnings for wrong student types
        self.student: DRLAgent = student

        self.state_dict_target = self.student.model.state_dict()
        self.state_dict_cache = None
        self.epsilon_scheduler = epsilon_scheduler
        self.double_q = double_q

    def is_target(self):
        return self.state_dict_cache is not None

    def swap_state_dicts(self):
        """
        Switches the regular DQN network parameters with the target networks parameters and vice versa, depending on
        its current status.
        """
        model = self.student.model
        if not self.is_target():
            self.state_dict_cache = deepcopy(model.state_dict())
            model.load_state_dict(self.state_dict_target)
        else:
            model.load_state_dict(self.state_dict_cache)
            self.state_dict_cache = None

    def run(
            self,
            n_epochs: int,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict] = None,
            batch_size: int = 4096,
            gamma: float = 0.99,
            memory_capacity: int = 50000,
            device: str = "cpu",
            seed: Union[int, np.random.Generator] = None,
    ):
        """
        Trains a reinforcement agent, acting according to the agents model
        or randomly (with exponentially decaying probability p_random)
        Each transition (state s, action a, next_state s', reward r) is stored in a memory.
        Each step in the environment is followed by a learning phase:
            a batch of memories is used to optimize the network model

        Parameters
        ----------
        optimizer_class
        optimizer_kwargs
        n_epochs
        batch_size
        gamma
        memory_capacity
        device
        seed
        """
        if not isinstance(self.student, DQNAgent):
            raise ValueError(
                f"Provided agent to train is not a DQN agent. Given: {type(self.student).__name__}"
            )
        rng = utils.rng_from_seed(seed)
        replays = Experience(memory_capacity, DQNMemory, rng)
        optimizer = optimizer_class(
            self.student.model.parameters(),
            **optimizer_kwargs if optimizer_kwargs else dict(),
        )
        for ep in range(n_epochs):
            self.game.reset()
            state = self.game.state
            state_tensor = self.student.state_to_tensor(state)
            status = self.game.state.status
            while status == Status.ongoing:
                policy = self.student.model(state_tensor)
                action = self.student.sample_action(
                    policy, self.epsilon_scheduler(ep), mode=PolicyMode.eps_greedy
                )
                move = self.action_map.action_to_move(action, state, self.student.team)

                # environment step for action
                status = self.game.run_step(move)
                reward = torch.tensor(
                    self.student.reward, dtype=torch.float, device=device
                )
                self.student.reward = 0

                # save transition as memory and optimize model

                next_state = (
                    self.student.state_to_tensor(state, perspective=self.student.team)
                    if status != Status.ongoing
                    else None
                )

                replays.push(
                    state, action, next_state, reward
                )  # store the transition in memory
                state = next_state  # move to the next state
                # one step of optimization of target network
                self.train(
                    optimizer,
                    replays.sample(batch_size),
                    gamma,
                    device,
                )

    def train(
            self,
            optimizer: torch.optim.Optimizer,
            batch: np.ndarray,
            gamma: float,
            device: str,
    ):
        """
        Sample batch from memory of environment transitions and train network to fit the
        temporal difference TD(0) Q-value approximation
        """
        model = self.student.model
        model.train()
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = []
        non_final_next_states = []

        batch_size = len(batch)

        states, actions, rewards = [], [], []
        for idx, entry in enumerate(batch):
            states.append(entry.state)
            actions.append(entry.action)
            rewards.append(entry.win)
            if (next_state := entry.next_state) is not None:
                non_final_mask.append(True)
                non_final_next_states.append(next_state)
            else:
                non_final_mask.append(False)

        non_final_mask = torch.ByteTensor(non_final_mask)
        non_final_next_states = torch.cat(non_final_next_states)

        state_batch = torch.cat(states)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        # Compute Q(s_t, a) - the model computes Q(s_t, . ), then we select the columns of actions taken
        q_values = model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states s_{t+1} of s_t.
        next_state_v = (
            torch.zeros(batch_size).float().to(device)
        )  # zero for terminal states

        # what would the model predict
        # computes argmax_a Q(s,a)
        estimated_q_values = model(non_final_next_states)
        estimated_best_choices = model(non_final_next_states).argmax(1)
        # now assign the double q estimates to the non-terminal states:
        if self.double_q:
            # assign double q learning q-values: Q_2( argmax_a Q_1(s,a))
            self.swap_state_dicts()
            next_state_v[non_final_mask] = model(non_final_next_states)[:, estimated_best_choices]
        else:
            # Otherwise, simply assign normal q-values: Q( argmax_a Q(s,a))
            next_state_v[non_final_mask] = estimated_q_values[:, estimated_best_choices]

        with torch.no_grad():
            # the expected next-state values require the stop-gradient!
            expected_q_values = (
                                                   next_state_v * gamma
                                           ) + reward_batch

        loss = F.smooth_l1_loss(
            q_values.view(-1), expected_q_values.view(-1)
        )  # compute Huber loss

        # optimize network
        optimizer.zero_grad()  # optimize towards expected q-values
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

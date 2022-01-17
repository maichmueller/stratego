from typing import Optional, Union, Type, Dict

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
from stratego.agent import DRLAgent
from stratego.core import ActionMap
from stratego.game import Game


class DQNAlgorithm(Algorithm):
    """
    A Deep Q-Network Teacher using the Double Q Learning method[1] if desired.
    This can also be used in conjunction with a dueling network architecture[2].

    References
    ----------
    [1] Van Hasselt, Hado, Arthur Guez, and David Silver.
        "Deep reinforcement learning with double q-learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 30. No. 1. 2016.
        https://ojs.aaai.org/index.php/AAAI/article/download/10295/10154
    [2] Wang, Ziyu, et al.
        "Dueling network architectures for deep reinforcement learning."
        International conference on machine learning.
        PMLR, 2016.
        http://proceedings.mlr.press/v48/wangf16.html
    [3] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves,
        A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A.,
        Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015).
        "Human-level control through deep reinforcement learning."
        Nature, 518(7540), 529–533.
        https://doi.org/10.1038/nature14236
    """

    def __init__(
            self,
            game: Game,
            student: DQNAgent,
            action_map: ActionMap,
            epsilon_scheduler: ExplorationScheduler,
            target_update_period: int = int(1e4),
            double_q: bool = True,
            model_folder: str = "./checkpoints/models",
            train_data_folder: str = "./checkpoints/data",
            seed: Union[int, np.random.Generator] = None,
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

        self.model = self.student.model
        self.target_model = deepcopy(self.student.model)
        self.epsilon_scheduler = epsilon_scheduler
        self.target_update_period: int = target_update_period
        self.rng = utils.rng_from_seed(seed)
        self._is_target: bool = False
        self._double_q: bool = double_q


    @property
    def is_target(self):
        return self._is_target

    @property
    def double_q(self):
        return self._double_q

    def update_target(self):
        """
        Switches the target DQN parameters with the training networks parameters.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def run(
            self,
            n_epochs: int = int(5e6),
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict] = None,
            batch_size: int = 32,
            gamma: float = 0.99,
            memory_capacity: int = 50000,
            device: str = "cpu",
    ):
        """
        Trains a reinforcement agent, acting according to the agents model
        or randomly (with exponentially decaying probability p_random)
        Each transition (state s, action a, next_state s', reward r) is stored in a memory.
        Each step in the environment is followed by a learning phase:
            a batch of memories is used to optimize the network model

        Parameters
        ----------
        n_epochs: int,
            the number of training epochs to run. Defaults to 5e6 [3].
        optimizer_class: Type[torch.optim.Optimizer],
            the optimizer class to use for training. Defaults to Adam.
        optimizer_kwargs: Dict[str, Any],
            the keyword arguments or the optimizer class.
        batch_size: int,
            the number of samples to use for the minibatch update in the game loop. Defaults to 32 [3].
        gamma: float,
            the discount factor for rewards.
        memory_capacity: int,
            the experience replay capacity.
        device: torch.device,
            the device to run the model on.
        """
        replays = Experience(memory_capacity, DQNMemory, self.rng)
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

                if ep % self.target_update_period == 0:
                    # copy over the current network into the target network
                    self.update_target()

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
        model = self.model
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

        # what would the training model predict?
        # Compute Q(s_t, a) - the model computes Q(s_t, . ), then we select the columns of actions taken in the sample
        q_values = model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states s_{t+1} of s_t.
        next_state_v = (
            torch.zeros(batch_size).float().to(device)
        )  # zero for terminal states (currently for all, but will be set further down)

        # what would the target model predict?
        # These values are the non-gradient carrying targets used as the goals for taining the model q-estimates.
        if self.double_q:
            # let Q_1 be the action selecting model (training model),
            #     Q_2 be the chosen action q-evaluating model (target model)
            # -> assign double q learning q-values: Q_2(s_{t+1}, argmax_a Q_1(s_{t+1},a))
            estimated_best_choices = model(non_final_next_states).argmax(1)
            next_state_v[non_final_mask] = self.target_model(non_final_next_states)[:, estimated_best_choices]
        else:
            # Otherwise, do standard Deep Q-learning, i.e. use DQN-algo q-values:
            #   Q(s, argmax_a Q(s,a, | theta_target) | theta_target)
            next_state_v[non_final_mask] = self.target_model(non_final_next_states).max(1)

        with torch.no_grad():
            # the expected q-values, which are the target values y_t, require the stop-gradient!

            # the q-values are computed via the current reward and discounted expected rewards (aka exp. state-values)
            # of the chosen action:
            # Q(s, a) = r + gamma * max_{a'} Q(s', a')
            # The value of next_state_v is 0 for any terminal state -> only the reward is used in this case
            expected_q_values = reward_batch + next_state_v * gamma

        loss = F.smooth_l1_loss(
            q_values.view(-1), expected_q_values.view(-1)
        )  # compute Huber loss

        # optimize network
        optimizer.zero_grad()  # optimize towards expected q-values
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

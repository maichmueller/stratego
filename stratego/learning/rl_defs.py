from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from collections import namedtuple
from functools import singledispatchmethod
from typing import Optional, Callable

import numpy as np


class RewardToken(Enum):
    illegal = 0  # punish illegal moves
    step = 1  # negative reward per agent step
    win = 2  # win game
    loss = 3  # lose game
    kill = 4  # kill enemy piece reward
    die = 5  # lose to enemy piece
    kill_mutually = 6  # mutual annihilation of attacking and defending piece


# For DQN: Stores a state-transition (s, a, s', r) tuple
DQNMemory = namedtuple("DQNMemory", "state, action, next_state, reward")
# For AlphaZero: Stores a state-policy-value tuple (s, pi, v, player) for the player
AlphaZeroMemory = namedtuple("AlphaZeroMemory", "state, pi, value, player")


class ReplayContainer:
    """
    Replay Memory for played states.
    Stores the content-tuple specified on creation.
    """

    def __init__(self, capacity, memory_class):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.memory_type = memory_class

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # only expand the memory if there is free capacity left
            self.memory.append(None)
        self.memory[self.position] = self.memory_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, rng_state: Optional[np.random.Generator] = None):
        if rng_state is not None:
            rng_state = np.random.default_rng()
        return rng_state.choice(self.memory, batch_size, replace=False)

    def extend(self, replays: ReplayContainer):
        for replay in replays:
            self.push(*replay)


class RandomActionScheduler(ABC):
    @abstractmethod
    def __call__(self, epoch: int) -> float:
        raise NotImplementedError


class LinearRandomActionScheduler(RandomActionScheduler):
    def __init__(
        self,
        start_epoch: int,
        end_epoch: int,
        linear_decline_stop: float = 0.1,
        exp_decay_rate: float = 0.1,
    ):
        self.start = start_epoch
        self.end = end_epoch
        self.slope = 1 / (start_epoch - end_epoch)
        self.intercept = self.end * (-self.slope)
        self.linear_decline_stop_ratio = linear_decline_stop
        self.linear_decline_cutoff = (1 / self.slope) * linear_decline_stop + end_epoch
        self.exp_decay_rate = exp_decay_rate

    def _linear_decline(self, epoch: int):
        return self.slope * epoch + self.intercept

    def _exp_finish(self, epoch: int):
        return (
            np.exp(-(epoch - self.linear_decline_cutoff) * self.exp_decay_rate)
            * self.linear_decline_stop_ratio
        )

    def __call__(self, epoch: int) -> float:
        if epoch < self.start:
            return 1.0
        if epoch > self.linear_decline_cutoff:
            return self._exp_finish(epoch)

        return self._linear_decline(epoch)

from __future__ import annotations
from enum import Enum

from collections import namedtuple
from functools import singledispatchmethod
from typing import Optional

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
DQNMemory = namedtuple("DQNTransition", "state, action, next_state, reward")
# For AlphaZero: Stores a state-policy-value tuple (s, pi, v, player) for the player
AlphaZeroMemory = namedtuple("AZTransition", "state, pi, value, player")


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


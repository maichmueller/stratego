from copy import deepcopy
from functools import singledispatchmethod

from stratego.agent import RLAgent

import numpy as np
import torch
from typing import Union, Iterable

from stratego.engine import Action, Logic, State, Team
from stratego.learning import PolicyMode


class DQNAgent(RLAgent):
    """
    Deep-Q Network agent. Estimates Q values in its model with double learning.
    """

    def __init__(self, *args, rng: Union[int, np.random.Generator] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(rng, np.random.Generator):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

    def decide_move(self, state: State, *args, **kwargs):
        if self.team == Team.red:
            state.flip_teams()

        self.model.to(self.device)  # no-op if already on correct device

        q_values = self.q_values(state)

        action = self.sample_action(q_values, mode=PolicyMode.greedy)
        if state.flipped_teams:
            state.flip_teams()
            action = self.action_map.invert_action(action)

        move = self.action_map.action_to_move(action, state, self.team)
        return move

    @singledispatchmethod
    def q_values(self, state):
        raise NotImplementedError

    @q_values.register
    def q_values(self, state: State):
        state_tensor = self.state_to_tensor(state)
        return self.model.predict(state_tensor)

    @q_values.register
    def q_values(self, state: torch.Tensor):
        return self.model.predict(state)



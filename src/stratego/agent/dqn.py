from copy import deepcopy
from functools import singledispatchmethod

from stratego.agent import DRLAgent

import numpy as np
import torch
from typing import Union, Iterable

from stratego.core import Action, Logic, State, Team
from stratego.learning import PolicyMode


class DQNAgent(DRLAgent):
    """
    Deep-Q Network agent..
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def q_values(self, state: Union[State, torch.Tensor]):
        if isinstance(state, State):
            state = self.state_to_tensor(state)
        return self.model.predict(state)

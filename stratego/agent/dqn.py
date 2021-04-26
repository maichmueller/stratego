from copy import deepcopy

from stratego.agent import AZAgent

import numpy as np
import torch
from typing import Union, Iterable

from stratego.engine import Action, Logic, State


class DQNAgent(AZAgent):
    """
    Deep-Q Network agent. Estimates policy and value with its model, but uses a prediction and target model to do so.
    The target model is updated only every C steps, as it is the 'fixed' reference model.
    """

    def __init__(self, *args, rng: Union[int, np.random.Generator] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dict_target = self.model.state_dict()
        self.state_dict_cache = None

        if not isinstance(rng, np.random.Generator):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

    def is_target(self):
        return self.state_dict_cache is not None

    def select_random_action(
            self,
            state: State,
            logic: Logic = Logic(),
    ) -> Action:
        action_mask = self.action_map.actions_mask(state.board, self.team, logic)
        return self.rng.choice(self.action_map.actions, p=action_mask / action_mask.sum())

    def swap_state_dicts(self):
        """
        Switches the regular DQN network parameters with the target networks parameters and vice versa, depending on
        its current status.
        """
        if not self.is_target():
            self.state_dict_cache = deepcopy(self.model.state_dict())
            self.model.load_state_dict(self.state_dict_target)
        else:
            self.model.load_state_dict(self.state_dict_cache)
            self.state_dict_cache = None

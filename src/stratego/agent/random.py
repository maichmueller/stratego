from .base import Agent
from ..core import State, Logic, Team

from typing import Union
import numpy as np


class RandomAgent(Agent):
    """
    Agent who chooses his actions at random
    """

    def __init__(
        self, team: Union[int, Team], seed: Union[np.random.Generator, int] = None
    ):
        super().__init__(team=team)
        if isinstance(seed, int):
            self.rng = np.random.default_rng(seed)
        elif isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            raise ValueError(
                f"Value of seed {type(seed)} not accepted as seed parameter."
            )

    def decide_move(self, state: State, logic: Logic = Logic()):
        all_moves = list(logic.possible_moves_iter(state.board, self.team))
        if not all_moves:
            return None
        else:
            return self.rng.choice(all_moves)

from stratego.agent import Agent
from stratego import Logic

import numpy as np

from stratego.state import State


class Random(Agent):
    """
    Agent who chooses his actions at random
    """

    def __init__(self, team):
        super(Random, self).__init__(team=team)

    def decide_move(self, state: State, *args, **kwargs):
        all_moves = list(Logic.possible_moves_iter(state.board, self.team))
        if not all_moves:
            return None
        else:
            return np.random.choice(all_moves)

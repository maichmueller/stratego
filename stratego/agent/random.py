from stratego.agent import Agent

import numpy as np

from stratego.engine import State, Logic


class RandomAgent(Agent):
    """
    Agent who chooses his actions at random
    """

    def __init__(self, team):
        super(RandomAgent, self).__init__(team=team)

    def decide_move(self, state: State, logic: Logic = Logic()):
        all_moves = list(logic.possible_moves_iter(state.board, self.team))
        if not all_moves:
            return None
        else:
            return np.random.choice(all_moves)

from stratego.agent import Agent


class Random(Agent):
    """
    Agent who chooses his actions at random
    """

    def __init__(self, team, setup=None):
        super(Random, self).__init__(team=team, setup=setup)


    def decide_move(self, state: State, judge: Judge):
        actions = utils.all_possible_moves(state.board, self.team)
        if not actions:
            return None
        else:
            return random.choice(actions)

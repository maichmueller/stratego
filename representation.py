from typing import Sequence

import numpy as np


class ActionRep:
    def __init__(self, game_dim: int):
        self.actors = None
        self.actions = None
        self.piecetype_to_actionrange = None
        self.action_dim = None
        self.game_dim = game_dim
        self.build_action_rep()

    def build_action_rep(self, available_types: Sequence[int]):

        action_rep_pieces = []
        action_rep_moves = []
        action_rep_dict = dict()
        for type_ in sorted(available_types):
            version = 1
            type_v = str(type_) + "_" + str(version)
            while type_v in action_rep_pieces:
                version += 1
                type_v = type_v[:-1] + str(version)
            if type_ in [0, 11]:
                continue
            elif type_ == 2:
                actions = (
                    [(i, 0) for i in range(1, self.game_dim)]
                    + [(0, i) for i in range(1, self.game_dim)]
                    + [(-i, 0) for i in range(1, self.game_dim)]
                    + [(0, -i) for i in range(1, self.game_dim)]
                )
                len_actions = len(actions)
                len_acts_sofar = len(action_rep_moves)
                action_rep_dict[type_v] = list(
                    range(len_acts_sofar, len_acts_sofar + len_actions)
                )
                action_rep_pieces += [type_v] * len_actions
                action_rep_moves += actions
            else:
                actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                action_rep_dict[type_v] = list(
                    range(len(action_rep_moves), len(action_rep_moves) + 4)
                )
                action_rep_pieces += [type_v] * 4
                action_rep_moves += actions
        self.piecetype_to_actionrange = action_rep_dict
        self.actions = tuple(action_rep_moves)
        self.actors = tuple(action_rep_pieces)
        self.action_dim = len(action_rep_moves)

"""
Pieces
"""
import numpy as np
from cythonized import utils


class Piece:
    def __init__(self, type, team, position):
        self.position = position
        # self.positions_history = [position]
        self.id = utils.set_id()
        self.potential_types = [0, 1, 2, 3, 10, 11]
        self.version = 1
        self.dead = False
        self.hidden = True
        self.guessed = False
        assert(type in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99))  # 0: flag, 11: bomb, 88: unknown, 99: obstacle
        self.type = type
        assert(team == 0 or team == 1 or team == 99)  # 99 is a neutral piece: e.g. obstacle
        self.team = team
        self.has_moved = False
        if type in (0, 11, 88, 99):
            self.can_move = False
            self.move_radius = 0
        elif type == 2:
            self.can_move = True
            self.move_radius = float('Inf')
        else:
            self.can_move = True
            self.move_radius = 1

    def __str__(self):  # for printing pieces on the board return type of piece
        return str(self.type)

    def __repr__(self):
        return f'{self.team}-{self.type}.{self.version}_{1*self.hidden}'

    def change_position(self, new_pos):
        # self.positions_history.append(new_pos)
        self.position = new_pos
        self.has_moved = True


"""
Pieces
"""
import numpy as np


class Piece:
    def __init__(self, type, team, position):
        self.position = position
        # self.positions_history = [position]
        self.unique_identifier = np.random.randint(0, 10000)
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
        if self.hidden:
            return "?"
        else:
            if self.type == 0:
                return "f"
            if self.type == 11:
                return "b"
            if self.type == 99:
                return "X"
            else:
                return str(self.type)

    def change_position(self, new_pos):
        #self.positions_history.append(new_pos)
        self.position = new_pos
        self.has_moved = True


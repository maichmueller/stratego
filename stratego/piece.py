from __future__ import annotations

from .spatial import Position
from typing import *


class Piece:
    # 0: flag, 11: bomb, 88: unknown, 99: obstacle
    _all_types = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99)

    def __init__(self, type_: int, team: int, position: Position, version: int = 1):
        self.position = position
        self.version = version
        self.dead = False
        self.hidden = True
        assert type_ in Piece._all_types, f"Piece type has to be one of {Piece._all_types}."
        self.type = type_
        assert (
            team == 0 or team == 1 or team == 99
        )  # 99 is a neutral piece: e.g. obstacle
        self.team = team
        self.has_moved = False
        if type_ in (0, 11, 88, 99):
            self.can_move = False
            self.move_radius = 0
        elif type_ == 2:
            self.can_move = True
            self.move_radius = float("Inf")
        else:
            self.can_move = True
            self.move_radius = 1

    def __str__(self):  # for printing pieces on the board
        return f"{self.type}.{self.version}"

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return f"{self.team}-{self.type}.{self.version}_{1*self.hidden}"

    def change_position(self, new_pos):
        self.position = new_pos
        self.has_moved = True

    def similar(self, other_piece: Piece) -> bool:
        """
        Return boolean indicating whether the passed piece is similar to this one.
        Similar is defined as same piece type and same version
        Parameters
        ----------
        other_piece: Piece,
            the other piece to compare with.

        Returns
        -------
        bool,
            is the other piece similar to this one.
        """
        return other_piece == self.type and other_piece == self.version

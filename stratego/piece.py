from __future__ import annotations

from functools import singledispatchmethod

from .game_defs import Team, Token
from .spatial import Position
from typing import *


class Piece:

    def __init__(self, token: Union[int, Token], team: int, position: Position, version: int = 1):
        self.team = Team(team)
        self.token = Token(token)
        self.version = version
        self.position = position
        self.dead = False
        self.hidden = True
        self.has_moved = False
        if self.token in (Token.flag, Token.bomb):
            self.can_move = False
            self.move_radius = 0
        elif self.token == Token.scout:
            self.can_move = True
            self.move_radius = float("Inf")
        else:
            self.can_move = True
            self.move_radius = 1

    def __str__(self):  # for printing pieces on the board
        return f"{self.token.value}.{self.version}"

    def __hash__(self):
        return hash((self.team, self.token, self.version))

    def __repr__(self):
        return f"{self.team}|[{self.token.value}.{self.version}]{'_H' if self.hidden else ''}"

    def change_position(self, new_pos):
        self.position = new_pos
        self.has_moved = True

    @singledispatchmethod
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
        raise NotImplementedError

    @similar.register(int)
    @similar.register(Token)
    def _(self, token: Union[int, Token], version: int):
        return Token(token) == self.token and version == self.version


@Piece.similar.register
def _(self, other: Piece):
    return self.similar(other.token, other.version)


class ShadowPiece:
    """
    Unknown Piece class. This is a placeholder for slicing the true board down to an individual agent's information.
    """

    def __init__(self, team: Team, position: Position):
        self.team = team
        self.position: Position = position


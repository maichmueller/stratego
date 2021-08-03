from __future__ import annotations

from .game_defs import Team, Token
from .position import Position

from abc import ABC
from functools import singledispatchmethod
from typing import *


class PieceBase(ABC):
    def __init__(
        self,
        position: Union[Position, Tuple[int, int]],
        hidden: bool = True,
        can_move: bool = True,
    ):
        self.position = Position(position)
        self.hidden = hidden
        self.can_move = can_move

    def change_position(self, new_pos):
        self.position = new_pos


class Piece(PieceBase):
    def __init__(
        self,
        position: Union[Position, Tuple[int, int]],
        team: Union[int, Team],
        token: Union[int, Token],
        version: int = 1,
        hidden: bool = True
    ):
        is_flag_or_bomb = token not in (Token.flag, Token.bomb)
        super().__init__(position, hidden, is_flag_or_bomb)
        self.team = Team(team)
        self.token = Token(token)
        self.version = version
        self.dead = False
        if is_flag_or_bomb:
            self.move_range = 0
        elif self.token == Token.scout:
            self.move_range = float("Inf")
        else:
            self.move_range = 1

    def __str__(self):  # for printing pieces on the board
        return f"{self.token.value}.{self.version}"

    def __hash__(self):
        return hash((self.team, self.token, self.version))

    def __repr__(self):
        return (
            f"{'B' if self.team == Team.blue else 'R'}"
            f"[{self.token.value}.{self.version}]"
            f"{'?' if self.hidden else ''}"
        )

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


@Piece.similar.register
def _(self, other: Piece):
    return self.similar(other.token, other.version)


@Piece.similar.register(int)
@Piece.similar.register(Token)
def _(self, token: Union[int, Token], version: int):
    return Token(token) == self.token and version == self.version


class Obstacle(PieceBase):
    def __init__(self, position: Union[Position, Tuple[int, int]]):
        super().__init__(Position(position), False, False)

    def change_position(self, new_pos):
        raise NotImplementedError("An obstacle can't move.")

    def __repr__(self):
        return "OBS"


class ShadowPiece(PieceBase):
    """
    Unknown Piece class. This is a placeholder for slicing the true board down to an individual agent's information.
    """
    @singledispatchmethod
    def __init__(self, position: Position, team: Team):
        super().__init__(position)
        self.team = team
        self.excluded_tokens: Set[Token] = set()

    @__init__.register
    def _(self, piece: Piece):
        self.__init__(piece.position, piece.team)

    def __repr__(self):
        return f"{'B' if self.team == Team.blue else 'R'}[?]"

    def exclude(self, *tokens: Token):
        self.excluded_tokens = self.excluded_tokens.union(tokens)



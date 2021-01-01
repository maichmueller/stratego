import copy
from typing import Sequence, Optional, Tuple, Dict, List

from .game_defs import Status, Token, Team
from .piece import Piece, ShadowPiece, Obstacle
from .position import Position, Move
from .board import Board

import numpy as np


class History:
    def __init__(self):
        self.turns: List[int] = []
        self.move: Dict[int, Move] = dict()
        self.team: Dict[int, Team] = dict()
        self.pieces: Dict[int, Tuple[Piece, Piece]] = dict()

    def commit_move(self, board: Board, move: Move, turn: int):
        """
        Commit the current move to history.
        """
        from_ = move.from_
        to_ = move.to_
        self.move[turn] = move
        self.pieces[turn] = copy.deepcopy(board[from_]), copy.deepcopy(board[to_])
        self.team[turn] = Team(turn % 2)


class State:
    def __init__(
        self,
        board: Board,
        starting_team: Team,
        piece_by_id_map: Dict[Tuple[Token, int, Team], Piece] = None,
        history: Optional[History] = None,
        move_count: int = 0,
        status: Status = Status.ongoing,
        canonical: bool = True,
        dead_pieces: Dict[Team, Dict[int, int]] = None,
    ):
        self.board = board
        self.piece_by_id: Dict[Tuple[Token, int, Team], Piece] = (
            piece_by_id_map
            if piece_by_id_map is not None
            else self._relate_piece_to_identifier(self.board)
        )
        self.history: History = history if history is not None else History()
        self.game_size: int = board.shape[0]

        self._is_canonical: bool = canonical

        self._turn_counter: int = move_count
        self.starting_team: Team = starting_team
        self.active_team: Team = Team((move_count + int(self.starting_team)) % 2)

        self.status: Status = status
        self.status_checked: bool = False

        if dead_pieces is not None:
            self.dead_pieces = dead_pieces
        else:
            self.dead_pieces = {Team(0): dict(), Team(1): dict()}

    def __str__(self):
        return np.array_repr(self.board)

    def __hash__(self):
        return hash(str(self))

    @property
    def turn_counter(self):
        return self._turn_counter

    @turn_counter.setter
    def turn_counter(self, count: int):
        self._turn_counter = count
        self.active_team = Team((count + int(self.starting_team)) % 2)

    def get_info_state(self, team: Team):
        return State(
            self.board.get_info_board(team),
            starting_team=self.starting_team,
            history=self.history,
            move_count=self.turn_counter,
            dead_pieces=self.dead_pieces,
        )

    def update_board(
        self,
        positions: Sequence[Position],
        pieces: Sequence[Optional[Piece]],
    ):
        """
        Parameters
        ----------
        positions: PositionType,
            piece board spatial
        pieces: Piece,
            the new piece at the spatial
        """
        for pos, piece in zip(positions, pieces):
            if piece is not None:
                piece.change_position(pos)
            self.board[pos] = piece
        self.status_checked = False
        return

    @staticmethod
    def _relate_piece_to_identifier(board: Board):
        piece_by_id = dict()
        for pos, piece in np.ndenumerate(board):
            if piece is not None and not isinstance(piece, ShadowPiece):
                id_tuple = piece.token, piece.version, piece.team
                piece_by_id[id_tuple] = piece

        return piece_by_id

    def force_canonical(self, player: Team):
        """
        Make the given player be team 0.
        :param player: int, the team to convert to
        """
        if player == 0 and self._is_canonical:
            # player 0 is still team 0
            return
        elif player == 1 and not self._is_canonical:
            # player 1 has already been made 0 previously
            return
        else:
            # flip team 0 and 1 and note the change in teams
            self._is_canonical = not self._is_canonical
            self.board = np.flip(self.board)
            for pos, piece in np.ndenumerate(self.board):
                # flip all team attributes
                if piece is not None and not isinstance(piece, Obstacle):
                    piece.team += 1
                    piece.position = pos

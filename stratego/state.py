import copy
from typing import Sequence, Optional, Tuple, Dict, List

from .game_defs import Status, Player, Token, Team
from .piece import Piece, ShadowPiece
from .spatial import Position, Move
from .spatial import Board

import numpy as np
from collections import defaultdict, Counter


class History:
    def __init__(self):
        self.turns: List[int] = []
        self.move: Dict[int, Move] = dict()
        self.team: Dict[int, int] = dict()
        self.pieces: Dict[int, Tuple[Piece, Piece]] = dict()

    def commit_move(self, board: Board, move: Move, player: player):
        """
        Commit the current move to history.
        """
        from_ = move.from_
        to_ = move.to_
        self.move[player] = move
        self.pieces[player] = copy.deepcopy(board[from_]), copy.deepcopy(board[to_])
        self.team[player] = player % 2


class State:
    def __init__(
        self,
        board: Board,
        piece_by_id_map: Dict[Tuple[Token, int, Team], Piece] = None,
        history: Optional[History] = None,
        move_count: int = 0,
        status: Status = Status.ongoing,
        canonical: bool = True,
        dead_pieces: Dict[Player, Dict[int, int]] = None,
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

        self._move_counter: int = move_count
        self.active_player: Player = Player(move_count)

        self.status: Status = status
        self.status_checked: bool = False

        if dead_pieces is not None:
            self.dead_pieces = dead_pieces
        else:
            self.dead_pieces = {Player(0): dict(), Player(1): dict()}

    def __str__(self):
        return np.array_repr(self.board)

    def __hash__(self):
        return hash(str(self))

    @property
    def move_counter(self):
        return self._move_counter

    @move_counter.setter
    def move_counter(self, count: int):
        self._move_counter = count
        self.active_player = Player(count)

    def get_info_state(self, player: Player):
        return State(
            self.board.get_info_board(player),
            history=self.history,
            move_count=self.move_counter,
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

    def force_canonical(self, player):
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
                if piece is not None and piece.team != 99:
                    piece.team ^= 1
                    piece.position = pos

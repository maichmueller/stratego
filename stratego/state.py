from typing import Sequence, Optional, Tuple, Dict

from .game_defs import Status, Player
from .piece import Piece
from .spatial import Position, Board

import numpy as np
from collections import defaultdict, Counter


class State:

    def __init__(
        self,
        board: Board,
        move_count: int = 0,
        dead_pieces: Dict[int, Dict[int, int]] = None,
    ):

        self.obstacle_positions = None
        if dead_pieces is not None:
            self.dead_pieces = dead_pieces
        else:
            self.dead_pieces = (dict(), dict())
        self.board = board
        self.game_size = board.shape[0]

        self.canonical_teams = True

        self._move_counter = move_count
        self.active_player = Player(move_count)

        self.terminal: Status = Status.ongoing
        self.terminal_checked = True

        if dead_pieces is not None:
            self.dead_pieces = dead_pieces
        else:
            self.dead_pieces = dict()

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
        self.terminal_checked = False
        return

    @staticmethod
    def check(piece, team, type_, version, hidden):
        if team == 0:
            if not hidden:
                # if it's about team 0, the 'hidden' status is unimportant
                return 1 * (piece.team == team and int == type_ and int == version)
            else:
                # hidden is only important for the single layer that checks for
                # only this quality!
                return 1 * (piece.team == team and piece.hidden == hidden)

        elif team == 1:
            # for team 1 we only get the info about type and version if it isn't hidden
            # otherwise it will fall into the 'hidden' layer
            if not hidden:
                if piece.hidden:
                    return 0
                else:
                    return 1 * (piece.team == team and int == type_ and int == version)
            else:
                return 1 * (piece.team == team and piece.hidden)
        else:
            # only obstace should reach here
            return 1 * (piece.team == team)

    def force_canonical(self, player):
        """
        Make the given player be team 0.
        :param player: int, the team to convert to
        """
        if player == 0 and self.canonical_teams:
            # player 0 is still team 0
            return
        elif player == 1 and not self.canonical_teams:
            # player 1 has already been made 0 previously
            return
        else:
            # flip team 0 and 1 and note the change in teams
            self.canonical_teams = not self.canonical_teams
            self.board = np.flip(self.board)
            for pos, piece in np.ndenumerate(self.board):
                # flip all team attributes
                if piece is not None and piece.team != 99:
                    piece.team ^= 1
                    piece.position = pos

    def check_status(self, flag_only=False, turn=0):
        if not any(self.dead_pieces):
            flags = sum([piece.team + 1 for piece in self.board.flatten() if piece is not None and piece.type == 0])
            if flags != 3:  # == 3 only if both flag 0 and flag 1 are present
                if flags == 1:  # agent 1 flag has been captured
                    self.terminal = 1  # agent 0 wins by flag
                else:
                    self.terminal = -1  # agent 1 wins by flag

        else:
            if self.dead_pieces[0][0] == 1:
                self.terminal = -1
            elif self.dead_pieces[1][0] == 1:
                self.terminal = 1

        if not flag_only:
            if not Logic.get_poss_moves(self.board, turn):
                self.terminal = (-1) ** (turn + 1) * 2  # agent of turn loses by moves
            elif not utils.get_poss_moves(self.board, (turn + 1) % 2):
                self.terminal = (-1) ** turn * 2  # agent of turn wins by moves

        if self.move_counter is not None and self.move_counter > self.max_nr_turns:
            self.terminal = 0

        self.terminal_checked = True

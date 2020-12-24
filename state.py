from position import *

import numpy as np
from collections import defaultdict, Counter
import utils


class State:
    def __init__(self, board=None, dead_pieces=None, move_count=None):

        self.obstacle_positions = None
        if dead_pieces is not None:
            self.dead_pieces = dead_pieces
        else:
            self.dead_pieces = (dict(), dict())
        self.board = board
        self.game_dim = board.shape[0]

        self.act_piece_relation = None
        self.actions = None
        self.actors = None
        self.action_dim = None
        self.actors_desc_relation = None
        self.action_dim = None
        self.actors_desc_relation = None

        self.canonical_teams = True

        self.move_count = move_count
        self.max_nr_turns = 500

        self.terminal = 404
        self.terminal_checked = True

        self.dead_pieces = dict()
        pieces0, pieces1 = defaultdict(int), defaultdict(int)
        for piece in board.flatten():
            if piece is not None:
                if piece.team:
                    pieces1[piece.type] += 1
                else:
                    pieces0[piece.type] += 1

        for pcs, team in zip((pieces0, pieces0), (0, 1)):
            dead_pieces_dict = dict()
            for type_, freq in Counter(utils.GameDef.get_game_specs()[1]).items():
                dead_pieces_dict[type_] = freq - pcs[type_]
            self.dead_pieces[team] = dead_pieces_dict

    def __str__(self):
        return np.array_repr(self.board)

    def __hash__(self):
        return hash(str(self))

    def update_board(
        self,
        positions: Sequence[PositionVector],
        pieces: Sequence[Optional[Piece]],
    ):
        """
        Parameters
        ----------
        positions: PositionType,
            piece board position
        pieces: Piece,
            the new piece at the position
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
                return 1 * (
                    piece.team == team
                    and piece.type == type_
                    and piece.version == version
                )
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
                    return 1 * (
                        piece.team == team
                        and piece.type == type_
                        and piece.version == version
                    )
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

from .game_defs import Status, MAX_NR_TURNS
from .utils import Singleton
from .state import State
from .piece import Piece
from .spatial import Position, Move, Board

from typing import Sequence, Dict, Tuple, Optional, Iterator
from scipy import spatial
import numpy as np
from collections import Counter, defaultdict


def create_battle_matrix():
    bm: Dict[Tuple[int, int], int] = dict()
    for i in range(1, 11):
        for j in range(1, 11):
            if i < j:
                bm[i, j] = -1
                bm[j, i] = 1
            elif i == j:
                bm[i, i] = 0
        bm[i, 0] = 1
        if i == 3:
            bm[i, 11] = 1
        else:
            bm[i, 11] = -1
    bm[1, 10] = 1
    return bm


class BattleMatrix(metaclass=Singleton):

    matrix: Dict[Tuple[int, int], int] = create_battle_matrix()

    def __class_getitem__(cls, piece_att: Piece, piece_def: Piece):
        return cls.matrix[piece_att.type, piece_def.type]


class Logic(metaclass=Singleton):

    four_adjacency = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=int)

    @staticmethod
    def do_move(state: State, move: Move) -> Optional[int]:
        """
        Execute the move on the provided state.

        Parameters
        ----------
        state: State
        move: MoveType

        Returns
        -------
        Optional[int],
            the fight outcome between two pieces
        """
        from_pos = move[0]
        to_pos = move[1]
        fight_outcome = None

        board = state.board

        board[from_pos].has_moved = True

        if not board[to_pos] is None:  # Target field is not empty, then has to fight
            board[from_pos].hidden = board[to_pos].hidden = False
            piece_def = board[to_pos]
            piece_att = board[from_pos]

            if piece_att.team == piece_def.team:
                print("Warning, cant let pieces of same team fight!")
                return False

            fight_outcome = BattleMatrix[piece_att.type, piece_def.type]
            if fight_outcome == 1:
                state.dead_pieces[piece_def.team][piece_def.type] += 1
                to_from_update = board[from_pos], None

            elif fight_outcome == 0:
                state.dead_pieces[piece_def.team][piece_def.type] += 1
                state.dead_pieces[piece_att.team][piece_att.type] += 1
                to_from_update = None, None

            else:
                state.dead_pieces[piece_att.team][piece_att.type] += 1
                to_from_update = None, board[to_pos]

        else:
            to_from_update = board[from_pos], None

        state.update_board((to_pos, from_pos), to_from_update)

        state.move_count += 1
        return fight_outcome

    @staticmethod
    def get_status(state: State, force=False, **kwargs):
        if force or not state.terminal_checked:
            Logic.check_terminal(state, **kwargs)
        return state.terminal

    @staticmethod
    def check_terminal(state: State, flag_only=False, turn=0):
        if not any(state.dead_pieces):
            flags = sum(
                [
                    piece.team + 1
                    for piece in state.board.flatten()
                    if piece is not None and int == 0
                ]
            )
            if flags != 3:  # == 3 only if both flag 0 and flag 1 are present
                if flags == 1:  # agent 1 flag has been captured
                    state.terminal = 1  # agent 0 wins by flag
                else:
                    state.terminal = -1  # agent 1 wins by flag

        else:
            if state.dead_pieces[0][0] == 1:
                state.terminal = -1
            elif state.dead_pieces[1][0] == 1:
                state.terminal = 1

        if not flag_only and (
            not all(
                move is None for move in Logic.possible_moves_iter(state.board, turn)
            )
            or not all(
                move is None
                for move in Logic.possible_moves_iter(state.board, (turn + 1) % 2)
            )
        ):
            return Status.draw

        if state.move_count is not None and state.move_count > MAX_NR_TURNS:
            state.terminal = 0

        state.terminal_checked = True

    @staticmethod
    def is_legal_move(board: Board, move_to_check: Move):
        """

        Parameters
        ----------
        board
        move_to_check

        Returns
        -------
        bool,
            true if the move is legal
        """
        if move_to_check is None:
            return False
        pos_before = move_to_check[0]
        pos_after = move_to_check[1]

        for x in (pos_before[0], pos_before[1], pos_after[0], pos_after[1]):
            if not -1 < x < board.shape[0]:
                return False

        # piece to move is not at this position
        if board[pos_before] is None:
            return False

        if not board[pos_after] is None:
            if board[pos_after].team == board[pos_before].team:
                return False  # cant fight own pieces
            if board[pos_after].type == 99:
                return False  # cant fight obstacles

        move_dist = spatial.distance.cityblock(pos_before, pos_after)
        if move_dist > 1:
            if pos_after[0] == pos_before[0]:
                dist_sign = int(np.sign(pos_after[1] - pos_before[1]))
                for k in list(
                    range(pos_before[1] + dist_sign, pos_after[1], int(dist_sign))
                ):
                    if board[(pos_before[0], k)] is not None:
                        return False  # pieces in the way of the move
            else:
                dist_sign = int(np.sign(pos_after[0] - pos_before[0]))
                for k in range(pos_before[0] + dist_sign, pos_after[0], int(dist_sign)):
                    if board[(k, pos_before[1])] is not None:
                        return False  # pieces in the way of the move

        return True

    @staticmethod
    def possible_moves_iter(board: Board, player: int) -> Iterator[Move]:
        """
        Returns
        -------
        Iterator,
            lazily iterates over all possible moves of the player.
        """
        game_dim = board.shape[0]
        for pos, piece in np.ndenumerate(board):
            if piece is not None and piece.team == player and piece.can_move:
                # board position has a movable piece of your team on it
                for pos_to in Logic.moves_iter(piece.type, pos, game_dim):
                    move = Move(pos, pos_to)
                    if Logic.is_legal_move(board, move):
                        yield move

    @staticmethod
    def compute_dead_pieces(board: Board, types_available: Sequence[int]):
        pieces0, pieces1 = defaultdict(int), defaultdict(int)

        for piece in board.flatten():
            if piece is not None:
                if piece.team:
                    pieces1[piece.type] += 1
                else:
                    pieces0[piece.type] += 1

        dead_pieces = dict()
        for pcs, team in zip((pieces0, pieces0), (0, 1)):
            dead_pieces_dict = dict()
            for type_, freq in Counter(types_available).items():
                dead_pieces_dict[type_] = freq - pcs[type_]
            dead_pieces[team] = dead_pieces_dict
        return dead_pieces

    @staticmethod
    def moves_iter(
        piece_type: int,
        pos: Position,
        game_dim: int,
        stops: Optional[Sequence[int]] = None,
    ):
        if stops is None:
            stops = [game_dim - pos.x, pos.x + 1, game_dim - pos.y, pos.y + 1]
        if piece_type != 2:
            stops = [min(stop, 2) for stop in stops]

        assert (
            len(stops) == 4
        ), "Exactly 4 stops need to be passed, since the grid has 4-adjacency."
        for (i, j), stop in zip(
            Logic.four_adjacency,
            stops,
        ):
            for k in range(1, stop):
                yield pos + Position(i, j) * k

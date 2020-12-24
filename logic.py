from utils import Singleton
from state import State
from pieces import Piece
from position import PositionType, MoveType

from typing import *
from scipy import spatial
import numpy as np


def create_battlematrix():
    bm = dict()
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

    matrix = create_battlematrix()

    def __class_getitem__(cls, piece_att: Piece, piece_def: Piece):
        return cls.matrix[piece_att.type, piece_def.type]


class Logic(metaclass=Singleton):
    @staticmethod
    def do_move(state: State, move: MoveType):
        """

        Parameters
        ----------
        state: State,
            the state on which to execute the move
        move: MoveType,
            the move to execute

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
    def is_terminal(state: State, force=False, **kwargs):
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
                    if piece is not None and piece.type == 0
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

        if not flag_only:
            if not Logic.all_possible_moves(state.board, turn):
                state.terminal = (-1) ** (turn + 1) * 2  # agent of turn loses by moves
            elif not Logic.all_possible_moves(state.board, (turn + 1) % 2):
                state.terminal = (-1) ** turn * 2  # agent of turn wins by moves

        if state.move_count is not None and state.move_count > state.max_nr_turns:
            state.terminal = 0

        state.terminal_checked = True

    @staticmethod
    def is_legal_move(board: np.ndarray, move_to_check: MoveType):
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
    def all_possible_moves(board: np.ndarray, team: int):
        """
        Returns
        -------
        List,
            possible actions for agent of team 'team'
        """
        game_dim = board.shape[0]
        actions_possible = []
        for pos, piece in np.ndenumerate(board):
            if piece is not None:  # board position has a piece on it
                if piece.team == team:
                    # check which moves are possible
                    if piece.can_move:
                        if piece.type == 2:
                            poss_fields = (
                                [
                                    (pos[0] + i, pos[1])
                                    for i in range(1, game_dim - pos[0])
                                ]
                                + [
                                    (pos[0], pos[1] + i)
                                    for i in range(1, game_dim - pos[1])
                                ]
                                + [(pos[0] - i, pos[1]) for i in range(1, pos[0] + 1)]
                                + [(pos[0], pos[1] - i) for i in range(1, pos[1] + 1)]
                            )
                            for pos_to in poss_fields:
                                move = (pos, pos_to)
                                if Logic.is_legal_move(board, move):
                                    actions_possible.append(move)
                        else:
                            poss_fields = [
                                (pos[0] + 1, pos[1]),
                                (pos[0], pos[1] + 1),
                                (pos[0] - 1, pos[1]),
                                (pos[0], pos[1] - 1),
                            ]
                            for pos_to in poss_fields:
                                move = (pos, pos_to)
                                if Logic.is_legal_move(board, move):
                                    actions_possible.append(move)
        return actions_possible

    @staticmethod
    def get_actions_mask(board, team, action_rep_dict, action_rep_moves):
        """
        :return: List of possible actions for agent of team 'team'
        """
        game_dim = board.shape[0]
        actions_mask = np.zeros(len(action_rep_moves), dtype=int)
        for pos, piece in np.ndenumerate(board):
            if (
                piece is not None and piece.team == team and piece.can_move
            ):  # board position has a piece on it
                # get the index range of this piece in the moves list
                p_range = np.array(
                    action_rep_dict[str(piece.type) + "_" + str(piece.version)]
                )
                # get the associated moves to this piece
                p_moves = [action_rep_moves[i] for i in p_range]
                if piece.type == 2:
                    poss_fields = (
                        [(pos[0] + i, pos[1]) for i in range(1, game_dim - pos[0])]
                        + [(pos[0], pos[1] + i) for i in range(1, game_dim - pos[1])]
                        + [(pos[0] - i, pos[1]) for i in range(1, pos[0] + 1)]
                        + [(pos[0], pos[1] - i) for i in range(1, pos[1] + 1)]
                    )

                    for pos_to in poss_fields:
                        move = (pos, pos_to)
                        if Logic.is_legal_move(board, move):
                            base_move = (pos_to[0] - pos[0], pos_to[1] - pos[1])
                            base_move_idx = p_moves.index(base_move)
                            actions_mask[p_range.min() + base_move_idx] = 1

                else:
                    poss_fields = [
                        (pos[0] + 1, pos[1]),
                        (pos[0], pos[1] + 1),
                        (pos[0] - 1, pos[1]),
                        (pos[0], pos[1] - 1),
                    ]
                    for pos_to in poss_fields:
                        move = (pos, pos_to)
                        if Logic.is_legal_move(board, move):
                            base_move = (pos_to[0] - pos[0], pos_to[1] - pos[1])
                            base_move_idx = p_moves.index(base_move)
                            actions_mask[p_range.min() + base_move_idx] = 1
        return actions_mask

    def mask_actions(board, team, action_rep_dict, action_rep_moves):
        """
        :return: List of possible actions for agent of team 'team'
        """
        game_dim = board.shape[0]
        actions_mask = np.zeros(len(action_rep_moves), dtype=int)
        for pos, piece in np.ndenumerate(board):
            if (
                piece is not None and piece.team == team and piece.can_move
            ):  # board position has a piece on it
                # get the index range of this piece in the moves list
                p_range = np.array(
                    action_rep_dict[str(piece.type) + "_" + str(piece.version)]
                )
                # get the associated moves to this piece
                p_moves = [action_rep_moves[i] for i in p_range]
                if piece.type == 2:
                    poss_fields = (
                        [(pos[0] + i, pos[1]) for i in range(1, game_dim - pos[0])]
                        + [(pos[0], pos[1] + i) for i in range(1, game_dim - pos[1])]
                        + [(pos[0] - i, pos[1]) for i in range(1, pos[0] + 1)]
                        + [(pos[0], pos[1] - i) for i in range(1, pos[1] + 1)]
                    )

                    for pos_to in poss_fields:
                        move = (pos, pos_to)
                        if is_legal_move(board, move):
                            base_move = (pos_to[0] - pos[0], pos_to[1] - pos[1])
                            base_move_idx = p_moves.index(base_move)
                            actions_mask[p_range.min() + base_move_idx] = 1

                else:
                    poss_fields = [
                        (pos[0] + 1, pos[1]),
                        (pos[0], pos[1] + 1),
                        (pos[0] - 1, pos[1]),
                        (pos[0], pos[1] - 1),
                    ]
                    for pos_to in poss_fields:
                        move = (pos, pos_to)
                        if is_legal_move(board, move):
                            base_move = (pos_to[0] - pos[0], pos_to[1] - pos[1])
                            base_move_idx = p_moves.index(base_move)
                            actions_mask[p_range.min() + base_move_idx] = 1
        return actions_mask

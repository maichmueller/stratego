from .game_defs import Status, MAX_NR_TURNS, Token, Team, BattleMatrix
from .state import State
from .position import Position, Move
from .board import Board
from stratego.utils import Singleton

from typing import Sequence, Dict, Tuple, Optional, Iterator, Union
from scipy import spatial
import numpy as np
from collections import Counter, defaultdict


class Logic(metaclass=Singleton):

    four_adjacency = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=int)

    @staticmethod
    def execute_move(state: State, move: Move) -> Optional[int]:
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

            fight_outcome = BattleMatrix[piece_att.token, piece_def.token]
            if fight_outcome == 1:
                state.dead_pieces[piece_def.team][piece_def.token] += 1
                to_from_update = board[from_pos], None

            elif fight_outcome == 0:
                state.dead_pieces[piece_def.team][piece_def.token] += 1
                state.dead_pieces[piece_att.team][piece_att.token] += 1
                to_from_update = None, None

            else:
                state.dead_pieces[piece_att.team][piece_att.token] += 1
                to_from_update = None, board[to_pos]

        else:
            to_from_update = board[from_pos], None

        state.update_board((to_pos, from_pos), to_from_update)

        state.turn_counter += 1
        return fight_outcome

    @staticmethod
    def undo_last_n_turns(state: State, n: int):
        """
        Undo the last n moves in the memory.

        state:  State,
            the state on which to undo the last rounds
        n:  int,
            number of moves to undo
        """
        for t in range(n):
            turn = state.history.turns[-1] - t
            move, (piece_from, piece_to) = (
                state.history.move[turn],
                state.history.pieces[turn],
            )
            state.board[move.from_] = piece_from
            state.board[move.to_] = piece_to

            if piece_to is not None:
                # if the target position held an actual piece, then there was a fight and
                # we need to update the dead pieces dictionary.
                if fight := BattleMatrix[piece_from.token, piece_to.token] == 0:
                    state.dead_pieces[Team(piece_from.team)][piece_from.token] -= 1
                    state.dead_pieces[Team(piece_to.team)][piece_to.token] -= 1
                elif fight == 1:
                    # the defender lost back then, so now remove it from the dead pieces
                    state.dead_pieces[Team(piece_to.team)][piece_to.token] -= 1
                else:
                    # the attacker lost back then, so now remove it from the dead pieces
                    state.dead_pieces[Team(piece_from.team)][piece_from.token] -= 1

    def get_status(self, state: State, force=False, **kwargs):
        if force or not state.status_checked:
            self.check_terminal(state, **kwargs)
        return state.status

    def check_terminal(self, state: State, flag_only=False, turn=0):
        for player in (Team(0), Team(1)):
            if state.dead_pieces[player][Token.flag] == 1:
                state.status = Status[player]

        if not flag_only:
            if state.turn_counter > MAX_NR_TURNS or any(
                not all(
                    move is None for move in self.possible_moves_iter(state.board, t)
                )
                for t in (turn, (turn + 1) % 2)
            ):
                state.status = Status.draw

        state.status_checked = True

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
            if board[pos_after].token == 99:
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

    def possible_moves_iter(
        self, board: Board, team: Union[int, Team]
    ) -> Iterator[Move]:
        """
        Returns
        -------
        Iterator,
            lazily iterates over all possible moves of the player.
        """
        team = Team(team)
        game_size = board.shape[0]
        for pos, piece in np.ndenumerate(board):
            if piece is not None and piece.team == team and piece.can_move:
                # board position has a movable piece of your team on it
                for move in self.moves_iter(piece.token, pos, game_size):
                    if Logic.is_legal_move(board, move):
                        yield move

    @staticmethod
    def compute_dead_pieces(board: Board, token_count: Sequence[int]):
        pieces0, pieces1 = defaultdict(int), defaultdict(int)

        for piece in board.flatten():
            if piece is not None:
                if piece.team:
                    pieces1[piece.token] += 1
                else:
                    pieces0[piece.token] += 1

        dead_pieces = dict()
        for pcs, team in zip((pieces0, pieces0), (0, 1)):
            dead_pieces_dict = dict()
            for token, freq in Counter(token_count).items():
                dead_pieces_dict[token] = freq - pcs[token]
            dead_pieces[team] = dead_pieces_dict
        return dead_pieces

    def moves_iter(
        self,
        token: Union[Token, int],
        pos: Position,
        game_size: int,
        distances: Optional[Sequence[int]] = None,
    ) -> Iterator[Move]:
        token = Token(token)

        if distances is None:
            distances = (game_size - pos.x, pos.x + 1, game_size - pos.y, pos.y + 1)
            if token.value != 2:
                distances = tuple(
                    map(
                        lambda x: min(x, 2),
                        distances,
                    )
                )

        else:
            assert (
                len(distances) == 4
            ), "A sequence of 4 distances needs to be passed, since the grid has 4-adjacency."

        for (i, j), stop in zip(
            self.four_adjacency,
            distances,
        ):
            for k in range(1, stop):
                yield Move(pos, pos + Position(i, j) * k)

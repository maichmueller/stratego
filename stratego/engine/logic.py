from .piece import Obstacle
from .game_defs import (
    Status,
    MAX_NR_TURNS,
    Token,
    Team,
    BattleMatrix,
    GameSpecification,
)
from .state import State
from .position import Position, Move
from .board import Board
from stratego.utils import Singleton

from typing import Sequence, Optional, Iterator, Union, Dict
from scipy import spatial
import numpy as np
from collections import Counter, defaultdict


class Logic(metaclass=Singleton):

    four_adjacency = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=int)

    @classmethod
    def execute_move(cls, state: State, move: Move) -> Optional[int]:
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
                print("Warning, can't let pieces of same team fight!")
                return False

            fight_outcome = BattleMatrix[piece_att.token, piece_def.token]
            board_update = dict()
            if fight_outcome == 1:
                # attacker won and moves onto defender spot, defender dies
                state.dead_pieces[piece_def.team][piece_def.token] += 1
                board_update.update({from_pos: board[from_pos], to_pos: None})

            elif fight_outcome == 0:
                # mutual annihilation, both disappear
                state.dead_pieces[piece_def.team][piece_def.token] += 1
                state.dead_pieces[piece_att.team][piece_att.token] += 1
                board_update.update({from_pos: None, to_pos: None})

            else:
                # attacker lost and dies, defender stays
                state.dead_pieces[piece_att.team][piece_att.token] += 1
                board_update.update({from_pos: None, to_pos: board[to_pos]})

        else:
            board_update = {from_pos: None, to_pos: board[from_pos]}

        state.update_board(board_update)

        state.turn_counter += 1

        return fight_outcome

    @classmethod
    def undo_last_n_turns(cls, state: State, n: int):
        """
        Undo the last n moves in the memory.

        state:  State,
            the state on which to undo the last rounds
        n:  int,
            number of moves to undo
        """
        for t in range(n):
            _, move, (piece_from, piece_to) = state.history.pop_last()

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

        state.unset_status()

    @classmethod
    def get_status(cls, state: State, force=False, **kwargs):
        if force or not state.status_checked:
            cls.check_terminal(state, **kwargs)
        return state.status

    @classmethod
    def check_terminal(cls, state: State, specs: GameSpecification):
        for player in (Team(0), Team(1)):
            dead_pieces = state.dead_pieces[player]
            if dead_pieces[Token.flag] == 1 or all(
                dead_pieces[token] == count
                for token, count in specs.token_count.items()
                if token not in [Token.flag, Token.bomb]
                # all move-able enemy pieces have been captured -> opponent won
            ):
                state.set_status(Status.win(player.opponent()))

        if state.turn_counter >= MAX_NR_TURNS or any(
            all(move is None for move in cls.possible_moves_iter(state.board, team))
            for team in (Team.blue, Team.red)
        ):
            state.set_status(Status.tie)

        # if no win or draw condition fits, then the match is ongoing
        state.set_status(Status.ongoing)

    @classmethod
    def is_legal_move(cls, board: Board, move_to_check: Move):
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
            if isinstance(board[pos_after], Obstacle):
                return False  # cant fight obstacles
            elif board[pos_after].team == board[pos_before].team:
                return False  # cant fight own pieces

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

    @classmethod
    def possible_moves_iter(
        cls, board: Board, team: Union[int, Team]
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
            if (
                piece is not None
                and not isinstance(piece, Obstacle)
                and piece.team == team
                and piece.can_move
            ):
                # board position has a movable piece of your team on it
                for move in cls.moves_iter(
                    piece.token, Position(tuple(map(int, pos))), game_size
                ):
                    if cls.is_legal_move(board, move):
                        yield move

    @classmethod
    def compute_dead_pieces(cls, board: Board, token_count: Dict[Token, int]):
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

    @classmethod
    def moves_iter(
        cls,
        token: Union[Token, int],
        pos: Position,
        game_size: int,
        distances: Optional[Sequence[int]] = None,
    ) -> Iterator[Move]:
        token = Token(token)

        if distances is None:
            distances = (game_size - pos.x, pos.x + 1, game_size - pos.y, pos.y + 1)
            if token != Token.scout:
                # if it isn't a scout, then it can move only one field
                # (which is bound by 2 due to range(2) stopping at 1)
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
            cls.four_adjacency,
            distances,
        ):
            for k in range(1, stop):
                yield Move(pos, pos + Position(i, j) * k)

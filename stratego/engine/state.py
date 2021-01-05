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

    def get_by_turn(self, turn: int):
        return self.team[turn], self.move[turn], self.pieces[turn]

    def get_by_index(self, idx: int):
        turn = self.turns[idx]
        return self.team[turn], self.move[turn], self.pieces[turn]

    def commit_move(self, board: Board, move: Move, turn: int):
        """
        Commit the current move to history.
        """
        from_ = move.from_
        to_ = move.to_
        self.move[turn] = move
        self.pieces[turn] = copy.deepcopy(board[from_]), copy.deepcopy(board[to_])
        self.team[turn] = Team(turn % 2)
        self.turns.append(turn)

    def pop_last(self):
        """
        Remove the latest entries from the history. Return the contents, that were removed.
        Returns
        -------
        tuple,
            all removed entries in sequence: turn, team, move, pieces
        """
        turn = self.turns[-1]
        return turn, self.team.pop(turn), self.move.pop(turn), self.pieces.pop(turn)


class State:
    def __init__(
        self,
        board: Board,
        starting_team: Team,
        piece_by_id_map: Dict[Tuple[Token, int, Team], Piece] = None,
        history: Optional[History] = None,
        turn_count: int = 0,
        flipped_teams: bool = False,
        status: Status = Status.ongoing,
        dead_pieces: Dict[Team, Dict[Token, int]] = None,
    ):
        self.board = board
        self.piece_by_id: Dict[Tuple[Token, int, Team], Piece] = (
            piece_by_id_map
            if piece_by_id_map is not None
            else self._relate_piece_to_identifier(self.board)
        )
        self.history: History = history if history is not None else History()
        self.game_size: int = board.shape[0]

        self.status: Status = status
        self.status_checked: bool = False

        self.flipped_teams: bool = flipped_teams

        self._turn_counter: int = turn_count
        self._starting_team: Team = starting_team
        self._active_team: Team = Team((turn_count + int(self._starting_team)) % 2)

        if dead_pieces is not None:
            self.dead_pieces = dead_pieces
        else:
            self.dead_pieces = {Team(0): dict(), Team(1): dict()}

    def __str__(self):
        return (
            f"{np.array_repr(self.board)}\n"
            f"Starting Team: {self._starting_team.name}\n"
            f"Active Team: {self._active_team.name}\n"
            f"Status: {self.status.name}\n"
            f"Turns: {str(self.turn_counter)}\n"
            f"Dead Pieces Blue: {self.dead_pieces[Team.blue]}\n"
            f"Dead Pieces Red: {self.dead_pieces[Team.red]}\n"
        )

    def __hash__(self):
        return hash(str(self))

    @property
    def active_team(self):
        # no outside manipulation intended
        return self._active_team

    @property
    def starting_team(self):
        # no outside manipulation intended
        return self._starting_team

    @property
    def turn_counter(self):
        return self._turn_counter

    @turn_counter.setter
    def turn_counter(self, count: int):
        self._turn_counter = count
        self._active_team = Team((count + int(self._starting_team)) % 2)

    def get_info_state(self, team: Team):
        return State(
            self.board.get_info_board(team),
            starting_team=self._starting_team,
            history=self.history,
            turn_count=self.turn_counter,
            flipped_teams=self.flipped_teams,
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
        positions: Position,
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

    def flip_teams(self):
        """
        Flip the teams and rotate the board, so that red becomes blue and vice versa on the board.
        This is important when an agent only understands himself to be a certain team,
        hence needs a canonical view of the state.
        """
        self.flipped_teams = not self.flipped_teams
        # flip team blue and red
        self._active_team = self._active_team.opponent()
        self._starting_team = self._starting_team.opponent()

        self.board = np.flip(self.board)
        # swap the dead pieces dictionary (note this is safe to do, due to execution order in python)
        self.dead_pieces[Team.blue], self.dead_pieces[Team.red] = (
            self.dead_pieces[Team.red],
            self.dead_pieces[Team.blue],
        )
        for pos, piece in np.ndenumerate(self.board):
            # flip all team attributes
            if piece is not None and not isinstance(piece, Obstacle):
                piece.team += 1
                piece.change_position(pos)

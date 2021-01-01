from .state import State
from .game_defs import Token, Team, GameSpecification
from .position import Position, Move
from .board import Board
from .logic import Logic
from .piece import Piece

from typing import Tuple, Dict, Union, List
from functools import singledispatchmethod
import numpy as np


class Action:
    def __init__(
        self, actor: Union[Tuple[Token, int], Piece], effect: Position
    ):
        self.actor: Tuple[Token, int]
        if isinstance(actor, tuple):
            self.actor = actor
        elif isinstance(actor, Piece):
            self.actor = actor.token, actor.version
        else:
            raise ValueError("Actor parameter needs to be either Tuple or Piece.")
        self.effect = effect

    def __call__(self, pos: Position):
        return pos + self.effect

    def __repr__(self):
        return f"{self.actor}: {self.effect}"

    def __hash__(self):
        return hash(self.actor)


class ActionMap:

    def __init__(self, game_specs: GameSpecification):
        self.specs = game_specs
        self.actions, self.actions_inverse = self._build_action_map(self.specs.token_count)
        self.action_dim = len(self.actions)

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError("Passed type not supported.")

    @__getitem__.register
    def _(self, arg: int):
        return self.actions[arg]

    @__getitem__.register
    def _(self, arg: tuple):
        return self.actions_inverse[arg]

    @__getitem__.register
    def _(self, arg: Piece):
        return self.actions_inverse[arg.token, arg.version]

    def _build_action_map(self, available_types: Dict[Token, int]):
        actions: List[Action] = []
        actions_inverse: Dict[Tuple[int, int], List[int]] = dict()
        for token, freq in available_types.items():
            if token in [Token.flag, Token.bomb]:
                continue

            moves: List[Move] = list(
                Logic.moves_iter(
                    token, Position(0, 0), self.specs.game_size, distances=[self.specs.game_size] * 4
                )
            )

            for version in range(1, freq + 1):
                la_before = len(actions)

                for (pos_before, pos_after) in moves:
                    actions.append(Action((token, version), pos_after - pos_before))

                la_after = len(actions)

                actions_inverse[actions[-1].actor] = list(range(la_before, la_after))

        self.actions_inverse = actions_inverse
        return actions, actions_inverse

    def actions_mask(self, board: Board, team: Union[Team, int]):
        """
        Get a boolean mask for the actions vector, masking out the illegal moves.

        Parameters
        ----------
        board: Board,
            the board on which the corresponding legal moves are calculated and hence illegal actions masked.
        team: Team or int,
            the team to which the action belongs.

        Returns
        -------

        """
        actions_mask = np.zeros(self.action_dim, dtype=np.int16)
        for (x, y), piece in np.ndenumerate(board):
            if (
                piece is not None and piece.team == Team(team) and piece.can_move
            ):  # board position has a piece on it
                pos = Position(x, y)
                # get the index range of this piece in the moves list
                actions_indices = self[piece]
                for action_idx in actions_indices:
                    action = self[action_idx]
                    if Logic.is_legal_move(board, Move(pos, action(pos))):
                        actions_mask[action_idx] = 1
        return actions_mask

    def action_to_move(self, idx: int, state: State, team: Team):
        """
        Converting an action index (0-action_dim) to a move, according to the action representation.

        Parameters
        ----------
        idx: int,
            index of the action in the action list.
        state: State,
            the state on which to translate the action to a move.
        team: Team,
            the team for which the action is meant.

        Returns
        -------
        Move
        """
        action = self.actions[idx]
        piece = state.piece_by_id[action.actor + (team, )]
        return Move(piece.position, action(piece.position))

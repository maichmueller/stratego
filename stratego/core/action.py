from __future__ import annotations

from .state import State
from .game_defs import Token, Team, GameSpecification
from .position import Position, Move
from .board import Board
from .logic import Logic
from .piece import Piece

from typing import Tuple, Dict, Union, List, Iterator
from functools import singledispatchmethod
import numpy as np


class Action:
    def __init__(self, actor: Union[Tuple[Token, int], Piece], effect: Position):
        # read only actor member
        self._actor: Tuple[Token, int]
        if isinstance(actor, tuple):
            assert len(actor) == 2, "'actor' parameter as tuple must have length 2."
            self._actor = actor
        elif isinstance(actor, Piece):
            self._actor = actor.token, actor.version
        else:
            raise ValueError("Actor parameter needs to be either Tuple or Piece.")
        # read only effect member
        self._effect: Position = effect
        # cached hash value.
        # Since no member variable is ever supposed to change,
        # it is valid to cache this value.
        self._hash = hash(self.actor + self.effect.coords)

    @property
    def effect(self):
        return self._effect

    @property
    def actor(self):
        return self._actor

    def __call__(self, pos: Position):
        return pos + self.effect

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"{self.actor}: {self.effect}"

    def __neg__(self):
        # invert the action (e.g. to represent its effect for the opposite player).
        return Action(self.actor, -self.effect)

    def __eq__(self, other: Action):
        return self.actor == other.actor and self.effect == other.effect


class ActionMap:
    def __init__(self, game_specs: GameSpecification):
        self.specs = game_specs
        (
            self.actions,  # type: List[Action]
            self.actions_inverse,  # type: Dict[Action, Action]
            self.actors_to_actions,  # type: Dict[Tuple[Token, int], List[Action]]
        ) = self._build_action_map(self.specs.token_count)
        self.action_dim = len(self.actions)

    def __len__(self):
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    @singledispatchmethod
    def __getitem__(self, arg):
        raise NotImplementedError("Passed type not supported.")

    @__getitem__.register
    def _(self, arg: int):
        """
        An integer is assumed to be merely the index of the action list
        """
        return self.actions[arg]

    @__getitem__.register
    def _(self, arg: tuple):
        """
        With a tuple we assume we are being given a tuple of (Token, Version)
        """
        return self.actors_to_actions[arg]

    @__getitem__.register
    def _(self, arg: Piece):
        return self.actors_to_actions[arg.token, arg.version]

    def invert_action(self, action: Action):
        """
        Get the action that does the inverse of the given action, e.g. when moving up by (1,0) the inverse is (-1, 0)
        """
        return self.actions_inverse[action]

    def _build_action_map(
        self, available_types: Dict[Token, int], logic: Logic = Logic()
    ):
        actions: List[Action] = []
        actions_inverse: Dict[Action, int] = dict()
        actors_to_actions: Dict[Tuple[Token, int], List[Action]] = dict()

        for token, freq in available_types.items():
            if token in [Token.flag, Token.bomb]:
                continue

            moves: Iterator[Move] = logic.moves_iter(
                token,
                Position(0, 0),
                self.specs.game_size,
                distances=[self.specs.game_size] * 4,
            )

            for version in range(1, freq + 1):
                actor = (token, version)
                actors_to_actions[actor] = []
                for (pos_before, pos_after) in moves:
                    action = Action((token, version), pos_after - pos_before)
                    actions.append(action)
                    actors_to_actions[actor].append(action)

            for idx, action in enumerate(actions):
                neg_action = -action
                for other_idx, other_action in enumerate(actions[idx + 1 :]):
                    if other_action == neg_action:
                        actions_inverse[action] = other_idx

        return actions, actions_inverse, actors_to_actions

    def actions_mask(
        self, board: Board, team: Union[Team, int], logic: Logic = Logic()
    ):
        """
        Get a boolean mask for the actions vector, masking out the illegal moves.

        Parameters
        ----------
        board: Board,
            the board on which the corresponding legal moves are calculated and hence illegal actions masked.
        team: Team or int,
            the team to which the action belongs.
        logic: Logic,
            the logic to use. Should not need to be used in general. Defaults to standard Stratego
            logic.


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
                    if logic.is_legal_move(board, Move(pos, action(pos))):
                        actions_mask[action_idx] = 1
        return actions_mask

    def action_to_move(
        self,
        action: Union[int, Action],
        identifier_to_piece: Dict[Tuple[Token, int, Team], Piece],
        team: Team,
    ):
        """
        Converting an action index (0-action_dim) to a move, according to the action representation.

        Parameters
        ----------
        action: Action or int,
            index of the action in the action list or the action itself.
        identifier_to_piece: Dict[Tuple[Token, int, Team], Piece],
            the mapping from piece identifier to the actual piece.
        team: Team,
            the team for which the action is meant.

        Returns
        -------
        Move
        """
        if isinstance(action, int):
            action = self.actions[action]
        piece = identifier_to_piece[action.actor + (team,)]
        return Move(piece.position, action(piece.position))

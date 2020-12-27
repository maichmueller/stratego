from .spatial import Position, Move
from .logic import Logic
from .piece import Piece

from typing import Sequence, Tuple, Dict, Union, List
from functools import singledispatchmethod
import numpy as np


class Action:
    def __init__(
        self, actor: Union[Tuple[int, int], Piece], effect: Position
    ):
        if isinstance(actor, tuple):
            self.actor = actor
        elif isinstance(actor, Piece):
            self.actor = actor.type, actor.version
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
    def __init__(self, all_types: Sequence[int], game_dim: int):
        self.game_dim = game_dim
        self.actions, self.actions_inverse = self._build_action_map(all_types)
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
        return self.actions_inverse[arg.type, arg.version]

    def _build_action_map(self, available_types: Sequence[int]):
        actions: List[Action] = []
        actions_inverse: Dict[Tuple[int, int], List[int]] = dict()
        for type_ in sorted(available_types):
            if type_ in [0, 11]:
                continue
            if actions:
                last_type, last_version = actions[-1].actor
                if type_ == last_type:
                    version = last_version + 1
                else:
                    version = 1
            else:
                version = 1

            la_before = len(actions)

            effects = list(
                Logic.moves_iter(
                    type_, Position(0, 0), self.game_dim, stops=[self.game_dim] * 4
                )
            )
            for effect in effects:
                actions.append(Action((type_, version), effect))

            la_after = len(actions)

            actions_inverse[actions[-1].actor] = list(range(la_before, la_after))

        self.actions_inverse = actions_inverse
        return actions, actions_inverse

    def actions_mask(self, board: np.ndarray, team: int):
        """
        :return: List of possible actions for agent of team 'team'
        """
        actions_mask = np.zeros(self.action_dim, dtype=np.int16)
        for (x, y), piece in np.ndenumerate(board):
            if (
                piece is not None and piece.team == team and piece.can_move
            ):  # board position has a piece on it
                pos = Position(x, y)
                # get the index range of this piece in the moves list
                actions_indices = self[piece]
                for action_idx in actions_indices:
                    action = self[action_idx]
                    if Logic.is_legal_move(board, Move(pos, action(pos))):
                        actions_mask[action_idx] = 1
        return actions_mask

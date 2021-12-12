from stratego.core import (
    Team,
    Piece,
    Obstacle,
    ShadowPiece,
    State,
    Token,
)

import torch
import numpy as np
from typing import Callable, List, Dict


def _build_default_filters(own_team: Team):
    """
    The default state representation is meant to fill boards with 0s and 1s depending on filters.
    This is meant to achieve layers of the following kind:
        Layer of own flag:
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        ...
        Layer of own hidden units:
            [[1, 0, 1, 1, 1],
             [1, 0, 1, 0, 1],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        ...

    Each layer has a callable checking whether the piece fulfills the condition.

    Parameters
    ----------
    own_team: Team,
        the team from whose perspective the filters are built.

    Returns
    -------
    List[Callable],
        a list of the filter functions used to create the filter layers.
    """
    filters = []

    # obstacle
    filters += lambda piece: isinstance(piece, Obstacle)

    # own pieces with versions as indicator
    for token in Token:
        filters += (
            lambda piece: (piece.version + 1) * isinstance(piece, Piece)
            and piece.team == own_team
            and not piece.hidden
            and piece.token == token
        )

    # all own hidden pieces
    filters += (
        lambda piece: isinstance(piece, Piece)
        and piece.team == own_team
        and piece.hidden
    )

    opponent = own_team.opponent()
    # opponent pieces
    for token in Token:
        # opponent units need to version specific filter
        # (since no actions correlate to a specific token version)
        filters += (
            lambda piece: isinstance(piece, Piece)
            and piece.team == opponent
            and not piece.hidden
            and piece.token == token
        )

    # all opponent hidden pieces
    filters += lambda piece: (
        not isinstance(piece, Obstacle) and piece.team == opponent
    ) and (isinstance(piece, ShadowPiece) or piece.hidden)

    return filters


class Representation:
    """
    A state Tensor representation class. Transforms a State object into a pytorch Tensor.
    The class should always be given the filters from both team perspectives.
    """

    def __init__(self, filters: Dict[Team, List[Callable]]):
        self.filters = filters
        self.state_rep_dim = len(self.filters)

    def __call__(self, state: State, perspective: Team = Team.blue) -> np.ndarray:
        """
        Convert the state object into a pytorch Tensor according to the filters.

        Parameters
        ----------
        state: State,
            the state to convert.
        perspective: Team,
            the team from whose perspective the conversion should be made.

        Returns
        -------
        np.ndarray,
            the converted state object as tensor.
        """
        raise NotImplementedError


class DefaultRepresentation(Representation):
    """
    The standard representation class to use the default filters for creating the state tensors.
    """

    def __init__(self):
        super().__init__(
            filters={
                Team.blue: _build_default_filters(Team.blue),
                Team.red: _build_default_filters(Team.red),
            },
        )

    def __call__(self, state: State, perspective: Team = Team.blue):
        board = state.board
        board_state = np.zeros(
            (1, self.state_rep_dim, board.shape[0], board.shape[1])
        )  # zeros for no information initially
        for i, check in enumerate(self.filters[perspective]):
            for pos, piece in np.ndenumerate(board):
                if check(piece):
                    board_state[(0, i) + pos.coords] = 1
        return board_state

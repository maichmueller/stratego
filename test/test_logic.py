import numpy as np

from stratego import Move
from stratego.core import Logic, Position, Team
from .build_board import minimal_state3
from .utils import eq_len


def test_moves_iter():
    positions = [Position(pos) for pos in [(4, 3), (4, 4), (0, 0), (2, 2)]]
    game_size = [5, 7, 10]

    for position in positions:
        for gs in game_size:
            moves = list(Logic().moves_iter(position, gs))
            moves_expected = [
                Move(position, Position(new_pos))
                for new_pos in [
                    (position.x, y) for y in list(range(0, gs)) if y != position.y
                ]
                + [(x, position.y) for x in list(range(0, gs)) if x != position.x]
            ]
            assert np.isin(moves, moves_expected).all() and eq_len(
                moves, moves_expected
            )


def test_legal_moves_iter():
    state = minimal_state3()
    moves_expected = {
        Team.blue: [
            Move()
        ],
        Team.red: [

        ]
    }
    moves = list(Logic().possible_moves_iter(state.board, Team.blue, state.config))
    moves_expected = []
    assert np.isin(moves, moves_expected).all() and eq_len(
        moves, moves_expected
    )

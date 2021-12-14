from stratego.core import (
    Logic,
    Team,
)
from build_board import minimal_state, minimal_state2


def test_logic():
    state = minimal_state()
    logic = Logic()

    moves_blue = list(logic.possible_moves_iter(state.board, Team.blue))
    moves_red = list(logic.possible_moves_iter(state.board, Team.red))
    x = 3

    state = minimal_state2()
    moves_blue = list(logic.possible_moves_iter(state.board, Team.blue))
    moves_red = list(logic.possible_moves_iter(state.board, Team.red))
    x = 3






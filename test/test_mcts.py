from stratego.algorithms import MCTS
from .build_board import minimal_state3
from stratego import ActionMap, rng_from_seed, Team, GameConfig


def test_mcts():
    state = minimal_state3()
    action_map = ActionMap(state.config)
    rng = rng_from_seed(0)
    mcts = MCTS(action_map, rng)
    mcts.search(state, Team.blue)

    assert True
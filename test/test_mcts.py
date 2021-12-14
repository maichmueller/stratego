from stratego.algorithms import MCTS
from build_board import minimal_state3
from stratego import ActionMap, rng_from_seed, Team


def test_mcts():
    state = minimal_state3()
    action_map = ActionMap(state.game_size)
    rng = rng_from_seed(0)
    mcts = MCTS(action_map, rng)
    mcts.search(state, Team.blue)

    assert True
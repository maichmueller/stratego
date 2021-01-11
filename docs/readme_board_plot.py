from stratego.agent import RandomAgent
from stratego.engine import Board, State, Team
from stratego import Game
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(112)
g = Game(RandomAgent(0, seed=rng), RandomAgent(1, seed=rng), game_size="s", seed=rng)

g.run_game(show=True, pause=0.2)
fig, ax = g.state.board.print_board(dpi=200)
plt.show()

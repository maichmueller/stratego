from stratego.agent import RandomAgent
from stratego.engine import Board, State, Team
from stratego import Game
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(115)
g = Game(RandomAgent(0, seed=rng), RandomAgent(1, seed=rng),
         game_size="l", seed=rng)

g.run_game(show=False, pause=0.1)
fig, ax = g.state.board.print_board(boardlength_in_inch=4, dpi=250)
fig.savefig("./images/game_example.png", transparent=True)
plt.show()

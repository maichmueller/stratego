from stratego import RandomAgent
from stratego import Game
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(115)
length = 4
g = Game(RandomAgent(0, seed=rng), RandomAgent(1, seed=rng),
         game_size="l", seed=rng)

g.run_game(show=False, pause=0.1)
fig, ax = g.state.board.print_board(figsize_sq=length, dpi=150)
fig.savefig("./images/game_example.png", transparent=True)
plt.show()

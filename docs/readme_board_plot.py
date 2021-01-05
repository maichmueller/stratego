from stratego.agent import RandomAgent
from stratego.engine import Board, State, Team
from stratego import Game
import matplotlib.pyplot as plt

g = Game(RandomAgent(0), RandomAgent(1), game_size="s", seed=1011)
g.state.board[(2, 3)] = g.state.board[(1, 3)]
g.state.board[(1, 3)] = None
g.state.board[(2, 3)].hidden = False
g.state.board[(3, 4)] = None
g.state.board[(3, 3)] = None
g.state.board[(2, 4)] = g.state.board[(4, 4)]
g.state.board[(4, 4)] = None
g.state.board[(1, 1)] = g.state.board[(3, 1)]
g.state.board[(1, 1)].hidden = False
g.state.board[(3, 1)] = None
fig, ax = g.state.board.print_board(dpi=200)
plt.show()

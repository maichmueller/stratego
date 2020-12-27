import game
import numpy as np
import piece
import agent


setup1 = np.full((5, 5), None)
setup1[(0,0)] = piece.Piece(0, 0, (0, 0))
setup2 = np.full((5, 5), None)
setup2[(0,4)] = piece.Piece(0, 1, (0, 4))


g = game.Game(agent.Random(0), agent.Random(1), fixed_setups=(setup1, setup2))
g.state.terminal_checked = False
print(g.state.get_status(turn=0))
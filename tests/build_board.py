from stratego.core import Board, Team, Piece, Obstacle, Token, State

import numpy as np


def minimal_state():
    board = Board(np.empty((5, 5), dtype=object))
    board[2, 2] = Obstacle((2, 2))
    board[0, 0] = Piece((0, 0), Team.blue, Token.scout)
    board[1, 1] = Piece((1, 1), Team.blue, Token.miner)
    board[4, 4] = Piece((4, 4), Team.red, Token.scout)
    board[3, 3] = Piece((3, 3), Team.red, Token.miner)
    state = State(board, Team.blue)
    return state


def minimal_state2():
    board = Board(np.empty((5, 5), dtype=object))
    board[2, 2] = Obstacle((2, 2))
    board[0, 0] = Piece((0, 3), Team.blue, Token.flag)
    board[1, 1] = Piece((0, 4), Team.blue, Token.bomb)
    board[1, 1] = Piece((3, 2), Team.blue, Token.miner)
    board[4, 4] = Piece((4, 0), Team.red, Token.flag)
    board[3, 3] = Piece((4, 1), Team.red, Token.bomb)
    board[3, 3] = Piece((1, 0), Team.red, Token.spy)
    state = State(board, Team.blue)
    return state

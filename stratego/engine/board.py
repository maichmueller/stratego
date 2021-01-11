from __future__ import annotations

from typing import Optional

from .game_defs import Token
from .piece import ShadowPiece, Piece, Team, Obstacle
from .position import Position

import numpy as np

from functools import singledispatchmethod
import matplotlib.pyplot as plt


class Board(np.ndarray):
    def __new__(cls, arr: np.ndarray, **kwargs):
        obj = super().__new__(
            cls, shape=arr.shape, buffer=arr, dtype=arr.dtype, **kwargs
        )
        return obj

    def print_board(
        self,
        figure: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        omniscient: bool = True,
        **kwargs
    ):
        """
        Plots the board in a pyplot figure.

        Parameters
        ----------
        omniscient
        figure: plt.Figure (optional),
            the figure of the plot. Is instantiated if not provided.
        ax: plt.Axes object (optional),
            the axis in which to plot the board. Is instantiated if not provided.
        kwargs: Dict,
            optional keyword arguments for plt.Figure
        """
        game_size = self.shape[0]
        if figure is None:
            figure = plt.figure(num=kwargs.pop("num", 42000), **kwargs)
        else:
            figure.clf()
        if ax is None:
            ax = figure.subplots(1, 1)

        # layout = np.add.outer(range(game_size), range(game_size)) % 2  # chess-pattern board
        layout = np.zeros((game_size, game_size))
        ax.imshow(
            layout, cmap=plt.cm.magma, alpha=0.0, interpolation="nearest"
        )  # plot board

        # plot lines separating each cell for visualization
        for i in range(game_size + 1):
            ax.plot(
                [i - 0.5, i - 0.5],
                [-0.5, game_size - 0.5],
                color="k",
                linestyle="-",
                linewidth=1,
            )
            ax.plot(
                [-0.5, game_size - 0.5],
                [i - 0.5, i - 0.5],
                color="k",
                linestyle="-",
                linewidth=1,
            )

        # go through all board positions and print the respective markers
        for pos in ((i, j) for i in range(game_size) for j in range(game_size)):
            piece = self[pos]  # select piece on respective board position
            # decide which marker type to use for piece
            if piece is not None:
                if isinstance(piece, Piece):
                    if piece.team == Team.red:
                        color = "r"  # red: player 1
                    elif piece.team == Team.blue:
                        color = "b"  # blue: player 0
                    else:
                        color = "k"  # black: obstacle

                    if piece.can_move:
                        form = "o"  # circle: for movable
                    else:
                        form = "s"  # square: either immovable or unknown piece
                    if piece.token == Token.flag:
                        form = "X"  # cross: flag

                    piece_marker = "".join(("-", color, form))
                    alpha = 0.3 if piece.hidden else 0.7
                    ax.plot(
                        pos[1], pos[0], piece_marker, markersize=37, alpha=alpha
                    )  # plot marker

                    if not piece.hidden or omniscient:
                        # token written on marker center
                        token_s, version_s = str(piece).split(".")
                        if not piece.hidden:
                            color = "black"
                        else:
                            color = "grey"
                        ax.annotate(
                            token_s,
                            xy=(pos[1], pos[0]),
                            color=color,
                            size=15,
                            ha="center",
                            va="center",
                        )
                        ax.annotate(
                            f"/{version_s}",
                            xy=(pos[1] + .25, pos[0] - .1),
                            color=color,
                            size=10,
                            ha="center",
                            va="center",
                        )
                elif isinstance(piece, Obstacle):
                    ax.plot(
                        pos[1], pos[0], "s", color="k", markersize=37, alpha=1
                    )  # plot marker

        # invert y makes numbering more natural.
        # Puts team blue on bottom, red on top
        ax.invert_yaxis()

        return figure, ax

    def get_info_board(self, player: Team) -> Board:
        """
        Slice the current board and extract the information board the given player can have,
        i.e. the board with all the information this player has.

        Parameters
        ----------
        player: Team,
            the player whose information is to be mapped

        Returns
        -------
        Board,
            the information board
        """
        info_board = Board(np.ndarray((5, 5), dtype=object))
        opponent = player.opponent()
        for piece in self.flatten():
            if piece is not None:
                if isinstance(piece, Obstacle) or piece.team != opponent:
                    info_board[piece.position] = piece
                else:
                    if piece.hidden:
                        info_board[piece.position] = ShadowPiece(
                            piece.position, opponent
                        )
                    else:
                        info_board[piece.position] = piece

        return info_board

    @singledispatchmethod
    def __getitem__(self, item):
        # whenever np.ndarray knows how to handle the type, we let it
        return super().__getitem__(item)

    @__getitem__.register(Position)
    def _(self, item: Position):
        # for our custom position type
        return super().__getitem__((item[0], item[1]))

    @singledispatchmethod
    def __setitem__(self, key, value):
        # whenever np.ndarray knows how to handle the type, we let it
        return super().__setitem__(key, value)

    @__setitem__.register(Position)
    def _(self, key: Position, value):
        # for our custom position type
        return super().__setitem__((key.x, key.y), value)

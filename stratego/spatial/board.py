from typing import Optional

from stratego.spatial import Position

import numpy as np

from functools import singledispatchmethod
import matplotlib.pyplot as plt


class Board(np.ndarray):
    def __new__(cls, arr: np.ndarray, **kwargs):
        obj = super().__new__(
            cls,
            shape=arr.shape,
            buffer=arr,
            dtype=arr.dtype,
            **kwargs
        )
        return obj

    def print_board(self, figure: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None):
        """
        Plots the board in a pyplot figure.

        Parameters
        ----------
        figure: plt.Figure (optional),
            the figure of the plot. Is instantiated if not provided.
        ax: plt.Axes object (optional),
            the axis in which to plot the board. Is instantiated if not provided.
        block: bool (def: false),
            whether to block any further execution after plotting.
        """
        game_dim = self.shape[0]
        # plt.interactive(False)  # make plot stay? true: close plot, false: keep plot
        if figure is None:
            figure = plt.Figure(42000)
        if ax is None:
            ax = figure.subplots(1, 1)

        ax.clf()
        # layout = np.add.outer(range(game_dim), range(game_dim)) % 2  # chess-pattern board
        layout = np.zeros((game_dim, game_dim))
        ax.imshow(
            layout, cmap=plt.cm.magma, alpha=0.0, interpolation="nearest"
        )  # plot board

        # plot lines separating each cell for visualization
        for i in range(game_dim + 1):
            plt.plot(
                [i - 0.5, i - 0.5],
                [-0.5, game_dim - 0.5],
                color="k",
                linestyle="-",
                linewidth=1,
            )
            plt.plot(
                [-0.5, game_dim - 0.5],
                [i - 0.5, i - 0.5],
                color="k",
                linestyle="-",
                linewidth=1,
            )

        # go through all board positions and print the respective markers
        for pos in ((i, j) for i in range(game_dim) for j in range(game_dim)):
            piece = self[pos]  # select piece on respective board position
            # decide which marker type to use for piece
            if piece is not None:
                # piece.hidden = False  # omniscient view

                if piece.team == 1:
                    color = "r"  # blue: player 1
                elif piece.team == 0:
                    color = "b"  # red: player 0
                else:
                    color = "k"  # black: obstacle

                if piece.can_move:
                    form = "o"  # circle: for movable
                else:
                    form = "s"  # square: either immovable or unknown piece
                if int == 0:
                    form = "X"  # cross: flag

                piece_marker = "".join(("-", color, form))
                alpha = 0.3 if piece.hidden else 1
                plt.plot(
                    pos[1], pos[0], piece_marker, markersize=37, alpha=alpha
                )  # plot marker
                # piece type written on marker center
                plt.annotate(
                    str(piece),
                    xy=(pos[1], pos[0]),
                    color="w",
                    size=20,
                    ha="center",
                    va="center",
                )
        # invert y makes numbering more natural; puts agent 1 on bottom, 0 on top !
        # plt.gca().invert_yaxis()

        return figure, ax

    @singledispatchmethod
    def __getitem__(self, item):
        return super(Board, self).__getitem__(item)

    @__getitem__.register
    def _(self, item: Position):
        return super(Board, self).__getitem__((item[0], item[1]))

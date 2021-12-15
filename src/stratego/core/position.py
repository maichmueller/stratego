from __future__ import annotations
from functools import singledispatchmethod
from typing import Sequence, Union, Tuple

import numpy as np


class Position:
    @singledispatchmethod
    def __init__(self, _):
        raise NotImplementedError

    @__init__.register(tuple)
    @__init__.register(list)
    @__init__.register(np.ndarray)
    def _(self, pos):
        assert len(pos) == 2, "Position sequence must have length 2."
        self.coords = tuple(pos)

    @__init__.register(int)
    @__init__.register(np.int16)
    @__init__.register(np.int32)
    @__init__.register(np.int64)
    def _(self, x, y):
        self.coords = x, y

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    def __abs__(self):
        # we're using the 1-norm as metric of |Position(x, y)|
        return abs(self.x) + abs(self.y)

    def __hash__(self):
        return hash(self.coords)

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __mul__(self, val: int):
        return Position(self.x * val, self.y * val)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return Position(self.x * (-1), self.y * (-1))

    def __truediv__(self, val: int):
        return self * (1 // val)

    def __getitem__(self, i: int):
        return self.coords[i]

    def __repr__(self):
        return str(self.coords)

    def __eq__(self, other: Position):
        return other.x == self.x and other.y == self.y


@Position.__init__.register(Position)
def _(self, pos: Position):
    self.coords = pos.x, pos.y


class Move:
    def __init__(
        self,
        pos1: Union[Tuple[int, int], Position],
        pos2: Union[Tuple[int, int], Position],
    ):
        self.from_to = (Position(pos1), Position(pos2))

    def __len__(self):
        diff = self.from_to[0] - self.from_to[1]
        return abs(diff.x) + abs(diff.y)

    @property
    def from_(self):
        return self.from_to[0]

    @property
    def to_(self):
        return self.from_to[1]

    def __getitem__(self, i: int):
        return self.from_to[i]

    def __repr__(self):
        return f"{str(self.from_)}->{str(self.to_)}"

    def __eq__(self, other: Move):
        return other.to_ == self.to_ and other.from_ == self.from_

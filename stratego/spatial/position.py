from __future__ import annotations
from functools import singledispatchmethod
from typing import Sequence, Union, Tuple, List


class Position:
    @singledispatchmethod
    def __init__(self, _):
        raise NotImplementedError

    @__init__.register(tuple)
    @__init__.register(list)
    def _(self, pos: Union[Tuple[int], List[int]]):
        assert len(pos) == 2, "Position sequence must have length 2."
        self.coords = pos

    @__init__.register(int)
    def _(self, x: int, y: int):
        self.coords = x, y

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

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


@Position.__init__.register(Position)
def _(self, pos: Position):
    self.coords = pos.x, pos.y


class Move:
    @singledispatchmethod
    def __init__(
        self,
        pos1: Union[Tuple[int, int], Position],
        pos2: Union[Tuple[int, int], Position],
    ):
        self.from_to = (Position(pos1), Position(pos2))

    @property
    def from_(self):
        return self.from_to[0]

    @property
    def to_(self):
        return self.from_to[1]

    def __getitem__(self, i: int):
        return self.from_to[i]


class Position:
    def __init__(self, x: int, y: int):
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


class Move:
    def __init__(self, pos1: Position, pos2: Position):
        self.from_to = [pos1, pos2]

    @property
    def from_(self):
        return self.from_to[0]

    @property
    def to_(self):
        return self.from_to[1]

    def __getitem__(self, i: int):
        return self.from_to[i]

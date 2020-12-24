
class PositionVector:
    def __init__(self, x: int, y: int):
        self.coords = x, y

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    def __add__(self, other):
        return PositionVector(self.x + other.x, self.y + other.y)

    def __mul__(self, val: int):
        return PositionVector(self.x * val, self.y * val)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, val: int):
        return self * (1 // val)

    def __getitem__(self, i: int):
        return self.coords[i]


class Move:
    def __init__(self, pos1: PositionVector, pos2: PositionVector):
        self.from_to = [pos1, pos2]

    @property
    def from_(self):
        return self.from_to[0]

    @property
    def to_(self):
        return self.from_to[1]

    def __getitem__(self, i: int):
        return self.from_to[i]

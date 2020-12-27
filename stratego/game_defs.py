from enum import Enum


class Status(Enum):
    ongoing = 404
    win_0 = 1
    win_1 = -1
    draw = 0


MAX_NR_TURNS = 500


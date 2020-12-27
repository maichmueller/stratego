
from enum import Enum
from functools import singledispatchmethod
from typing import Union


class Player:
    class Team(Enum):
        blue = 0
        red = 1

    def __init__(self, team: Union[bool, int]):
        team = int(team)
        self._team: Player.Team = Player.Team(team % 2)

    @singledispatchmethod
    def __add__(self, value):
        raise NotImplementedError

    @__add__.register
    def _(self, value: bool):
        return Player.Team(self._team.value + value)

    @__add__.register
    def _(self, value: int):
        return Player.Team((self._team.value + value) % 2)


class Status(Enum):
    ongoing = 404
    win_blue = 1
    win_red = -1
    draw = 0


MAX_NR_TURNS = 500


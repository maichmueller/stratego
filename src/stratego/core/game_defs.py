from __future__ import annotations

from abc import ABC
from enum import Enum
from functools import singledispatchmethod
from typing import Union, Tuple, Dict, Sequence, Iterable


class Team(Enum):
    blue = 0
    red = 1

    def __int__(self):
        return self.value

    @singledispatchmethod
    def __add__(self, value):
        raise NotImplementedError

    @__add__.register
    def _(self, value: int):
        return Team((self.value + value) % 2)

    def opponent(self):
        return self + 1


class Token(Enum):
    flag = 0
    spy = 1
    scout = 2
    miner = 3
    sergeant = 4
    lieutenant = 5
    captain = 6
    major = 7
    colonel = 8
    general = 9
    marshall = 10
    bomb = 11
    obstacle = 99

    def __int__(self):
        return self.value


class HookPoint(Enum):
    pre_run = 0
    post_run = 1
    pre_move_decision = 2
    post_move_decision = 3
    pre_move_execution = 4
    post_move_execution = 5


class Status(Enum):
    ongoing = 404
    win_blue = 1.
    win_red = -1.
    tie = 0.

    @singledispatchmethod
    @classmethod
    def win(cls, team: int):
        raise NotImplementedError

    @win.register(int)
    @win.register(Team)
    @classmethod
    def _(cls, team: Union[int, Team]):
        team = Team(team)
        if team == Team.red:
            return Status.win_red
        if team == Team.blue:
            return Status.win_blue

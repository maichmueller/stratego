from __future__ import annotations

from abc import ABC
from enum import Enum
from functools import singledispatchmethod
from typing import Union, Tuple, Dict


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
    win_blue = 1
    win_red = -1
    draw = 0

    @singledispatchmethod
    def __class_getitem__(cls, team: int):
        raise NotImplementedError

    @__class_getitem__.register(int)
    @__class_getitem__.register(Team)
    def _(self, team: Union[int, Team]):
        team = Team(team)
        if team == Team.red:
            return Status.win_red
        if team == Team.blue:
            return Status.win_blue


MAX_NR_TURNS = 500


def _create_battle_matrix():
    bm: Dict[Tuple[Token, Token], int] = dict()
    for i in Token:
        for j in Token:
            if any(k in [Token.flag, Token.bomb] for k in (i, j)):
                continue
            if i.value < j.value:
                bm[i, j] = -1
                bm[j, i] = 1
            elif i == j:
                bm[i, i] = 0
        bm[i, Token.flag] = 1
        if i == Token.miner:
            bm[i, Token.bomb] = 1
        else:
            bm[i, Token.bomb] = -1
    bm[Token.spy, Token.marshall] = 1
    return bm


class BattleMatrix(ABC):

    matrix = _create_battle_matrix()

    def __class_getitem__(cls, token_att: Token, token_def: Token):
        return cls.matrix[token_att, token_def]


def build_specs(game_size: int):
    if game_size == 5:
        token_count = {
            Token.flag: 1,
            Token.spy: 1,
            Token.scout: 3,
            Token.miner: 2,
            Token.marshall: 1,
            Token.bomb: 2,
        }
        obstacle_positions = [(2, 2)]
        setup_rows = {Team.blue: [0, 1], Team.red: [3, 4]}
        game_size = 5
    elif game_size == 7:
        token_count = {
            Token.flag: 1,
            Token.spy: 1,
            Token.scout: 5,
            Token.miner: 3,
            Token.sergeant: 3,
            Token.lieutenant: 2,
            Token.captain: 1,
            Token.marshall: 1,
            Token.bomb: 4,
        }
        obstacle_positions = [(3, 1), (3, 5)]
        setup_rows = {Team.blue: [0, 1, 2], Team.red: [4, 5, 6]}
        game_size = 7
    elif game_size == 10:
        token_count = {
            Token.flag: 1,
            Token.spy: 1,
            Token.scout: 8,
            Token.miner: 5,
            Token.sergeant: 4,
            Token.lieutenant: 4,
            Token.captain: 4,
            Token.major: 3,
            Token.colonel: 2,
            Token.general: 1,
            Token.marshall: 1,
            Token.bomb: 6,
        }
        obstacle_positions = [
            (4, 2),
            (5, 2),
            (4, 3),
            (5, 3),
            (4, 6),
            (5, 6),
            (4, 7),
            (5, 7),
        ]
        setup_rows = {Team.blue: [0, 1, 2, 3], Team.red: [6, 7, 8, 9]}
        game_size = 10
    else:
        raise ValueError(f"Board size {game_size} not supported.")
    return token_count, obstacle_positions, setup_rows, game_size


_game_specs = {size: build_specs(size) for size in (5, 7, 10)}


class GameSpecification:
    def __init__(self, game_size: Union[str, int]):
        if isinstance(game_size, str):
            game_size = game_size.lower()
            if game_size in ["s", "small"]:
                game_size = 5
            elif game_size in ["m", "medium"]:
                game_size = 7
            elif game_size in ["l", "large"]:
                game_size = 10
            else:
                raise ValueError(f"Game size {game_size} not supported.")
        elif isinstance(game_size, int):
            assert game_size in [
                5,
                7,
                10,
            ], "Integer 'game_size' parameter must be one of [5, 7, 10]."
        self._game_specs = _game_specs[game_size]

    @property
    def token_count(self):
        return self._game_specs[0]

    @property
    def obstacle_positions(self):
        return self._game_specs[1]

    @property
    def setup_rows(self):
        return self._game_specs[2]

    @property
    def game_size(self):
        return self._game_specs[3]

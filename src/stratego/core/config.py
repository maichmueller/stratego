from abc import ABC
from typing import Tuple, Union, Dict, Iterable, Sequence, Optional, List, Type

from .position import Position
from .game_defs import Token, Team


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


def _verify_game_size(game_size):
    if isinstance(game_size, str):
        game_size = game_size.lower()
        if game_size in ["s", "small"]:
            game_size = 5
        elif game_size in ["m", "medium"]:
            game_size = 7
        elif game_size in ["l", "large"]:
            game_size = 10
        else:
            raise ValueError(
                f"'game_size' parameter as string must be one of\n"
                f"\t['s', 'small', 'm', 'medium', 'l', 'large']."
            )
    elif isinstance(game_size, int):
        assert game_size in [5, 7, 10,], (
            "'game_size' parameter as integer must be one of\n" "\t[5, 7, 10]."
        )
    return game_size


def _verify_setups(
    setups: Tuple[Dict[Position, Token]], setup_rows: Dict[Team, Sequence[int]]
):
    for i, setup in enumerate(setups):
        rows = setup_rows[Team(i)]
        for pos, token in setup.items():
            if pos.x not in rows:
                raise ValueError(
                    "Passed setup parameter not valid for game configuration."
                )


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

    def __class_getitem__(cls, tokens: Sequence[Token]):
        return cls.matrix[tokens[0], tokens[1]]


class GameConfig:
    def __init__(
        self,
        game_size: Union[str, int],
        starting_team: Team,
        setups: Optional[Tuple[Dict[Position, Token]]] = None,
        max_nr_turns: int = 500,
        battlematrix: Type[BattleMatrix] = BattleMatrix
    ):
        game_size = _verify_game_size(game_size)
        self._game_specs = _game_specs[game_size]
        self._starting_team = starting_team
        if setups is not None:
            _verify_setups(setups, self.setup_rows)
        self._setups = setups
        self._max_nr_turns = max_nr_turns
        self._battle_matrix = battlematrix

    @property
    def token_count(self) -> Dict[Token, int]:
        return self._game_specs[0]

    @property
    def obstacle_positions(self) -> Sequence[Position]:
        return self._game_specs[1]

    @property
    def setup_rows(self) -> Dict[Team, Sequence[int]]:
        return self._game_specs[2]

    @property
    def game_size(self) -> int:
        return self._game_specs[3]

    @property
    def setups(self) -> Tuple[Dict[Position, Token]]:
        return self._setups

    @property
    def starting_team(self) -> Team:
        return self._starting_team

    @property
    def max_nr_turns(self) -> int:
        return self._max_nr_turns

    @property
    def battlematrix(self) -> Type[BattleMatrix]:
        return self._battle_matrix

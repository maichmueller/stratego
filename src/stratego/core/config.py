from abc import ABC
from typing import Tuple, Union, Dict, Iterable, Sequence, Optional, List, Type

from .position import Position
from .game_defs import Token, Team
from collections import Counter, namedtuple

Specification = namedtuple(
    "Specification", ["token_counts", "obstacle_pos", "setup_rows", "game_size"]
)


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
        token_count = {team: token_count for team in Team}
        game_size = 10
    else:
        raise ValueError(f"Board size {game_size} not supported.")
    return Specification(token_count, obstacle_positions, setup_rows, game_size)


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
    setups: Dict[Team, Dict[Position, Token]], setup_rows: Dict[Team, Sequence[int]]
):
    for team, setup in setups.items():
        rows = setup_rows[team]
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


def _create_move_ranges():
    mr = {}
    for token in Token:
        if token == Token.scout:
            mr[token] = float("inf")
        elif token in [Token.flag, Token.bomb]:
            mr[token] = 0
        else:
            mr[token] = 1
    return mr


class MoveRange(ABC):

    lookup = _create_move_ranges()

    def __class_getitem__(cls, token: Token):
        return cls.lookup[token]


def _update_token_counts(game_specs: Specification, setups):
    token_counts = {}
    for team, setup in setups.items():
        token_counts[team] = Counter(setup.values())
    return Specification(
        token_counts=token_counts,
        game_size=game_specs.game_size,
        obstacle_pos=game_specs.obstacle_pos,
        setup_rows=game_specs.setup_rows,
    )


class GameConfig:
    def __init__(
        self,
        game_size: Union[str, int],
        starting_team: Team,
        *,  # the remainder should be kw-only arguments
        setups: Optional[Dict[Team, Dict[Position, Token]]] = None,
        max_nr_turns: int = 500,
        battlematrix: Type[BattleMatrix] = BattleMatrix,
        move_ranges: Type[MoveRange] = MoveRange,
        allow_atypical_setup: bool = False,
    ):
        game_size = _verify_game_size(game_size)
        self._game_specs: Specification = _game_specs[game_size]
        self._starting_team = starting_team
        if setups is not None:
            if not allow_atypical_setup:
                _verify_setups(setups, self.setup_rows)
            self._game_specs = _update_token_counts(self._game_specs, setups)
        self._setups = setups
        self._max_nr_turns = max_nr_turns
        self._battle_matrix = battlematrix
        self._move_ranges = move_ranges

    @property
    def token_count(self) -> Dict[Team, Dict[Token, int]]:
        return self._game_specs.token_counts

    @property
    def obstacle_positions(self) -> Sequence[Position]:
        return self._game_specs.obstacle_pos

    @property
    def setup_rows(self) -> Dict[Team, Sequence[int]]:
        """
        The default setup rows for this game configuration.
        These are the rows one could choose from upon redrawing a setup.
        """
        return self._game_specs.setup_rows

    @property
    def game_size(self) -> int:
        return self._game_specs.game_size

    @property
    def setups(self) -> Dict[Team, Dict[Position, Token]]:
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

    @property
    def move_ranges(self) -> Type[MoveRange]:
        return self._move_ranges

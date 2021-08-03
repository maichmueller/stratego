from typing import Dict, List, Callable, Optional, Union, Iterable

import numpy as np
import copy

# from collections import Counter

# from scipy import optimize

import torch
from collections import defaultdict
from abc import ABC

from stratego.core import (
    State,
    Logic,
    Team,
    HookPoint,
    GameSpecification,
    Board,
    Move,
    Piece,
    ActionMap,
    Action,
    InfoBoard,
    ShadowPiece,
    Token,
)
from src.stratego.learning import (
    RewardToken,
    Representation,
    PolicyMode,
)
from .. import utils


class Agent(ABC):
    """
    A general abstract agent base class.
    """

    def __init__(
        self,
        team: Union[int, Team]
    ):
        self.team = Team(team)
        self.rng = None
        self.hooks: Dict[HookPoint, List[Callable]] = defaultdict(list)

    def set_rng(self, seed: Union[int, np.random.RandomState, np.random.Generator] = None):
        self.rng = utils.rng_from_seed(seed)
        return self

    def setup(self, info_board: InfoBoard):
        pass

    def decide_move(self, state: State, logic: Logic = Logic()) -> Move:
        """
        Decide the move to make for the given state of the game.

        Parameters
        ----------

        state: State,
            the state on which the decision is to be made.
        logic: Logic,
            the logic to use in the core. Can be changed to vary the game mode if desirable.

        Returns
        -------
        Move,
            the chosen move to make on the state.
        """
        raise NotImplementedError

    def register_hooks(self, hooks_map: Dict[HookPoint, List[Callable]]):
        """
        Add hooks which are supposed to be executed at specific positions during the game run.
        """
        for hook_point, hooks in hooks_map.items():
            self.hooks[hook_point].extend(hooks)


class RLAgent(Agent, ABC):
    """
    Reinforcement Learning agent
    """

    def __init__(
        self,
        team: Union[int, Team],
        action_map: ActionMap,
        representation: Representation,
        reward_map: Dict[RewardToken, float]
    ):
        super().__init__(team=team)
        self.action_map = action_map
        self.action_dim = len(action_map.actions)
        self.representation = representation

        self.reward = 0
        self.reward_map: Dict[RewardToken, float] = reward_map

    def sample_action(
        self,
        policy: Union[torch.Tensor, np.ndarray, Iterable[float]],
        *args,
        mode: PolicyMode,
        **kwargs,
    ) -> Action:
        """
        Choose the action to take with which to form the next move.
        This is usually just the argmax of policies, but may be overwritten.
        Any args or kwargs are passed down to the `_select_action` implementation.

        Parameters
        ----------
        policy: torch.Tensor, np.ndarray or Iterable[float],
            the policy from which to select the appropriate action.

        mode: PolicyMode,
            the value corresponds to different action selection paradigms
                stochastic - sampled according to policy
                epsilon-greedy - argmax selected with prob 1-eps, rest share remaining eps prob. mass
                greedy - argmax of policy (deterministic)
                uniform - samples from all actions with p > 0 uniformly

        Returns
        -------
        Action,
            the chosen action.
        """
        if mode == PolicyMode.stochastic:
            return self._select_action_stochastic(policy, *args, **kwargs)
        elif mode == PolicyMode.eps_greedy:
            eps = float(args[0])
            return self._select_action_eps_greedy(policy, eps, *args, **kwargs)
        elif mode == PolicyMode.greedy:
            return self._select_action_greedy(policy, *args, **kwargs)
        elif mode == PolicyMode.uniform:
            return self._select_action_uniform(*args, **kwargs)
        else:
            raise ValueError(
                f"mode {mode} not supported. Allowed values are {[e.value for e in PolicyMode]}."
            )

    def _select_action_greedy(
        self, policy: Union[torch.Tensor, np.ndarray, Iterable[float]], *args, **kwargs
    ):
        """
        Select action deterministically via argmax by default.
        """
        return self.action_map[int(torch.argmax(torch.tensor(policy)))]

    def _select_action_stochastic(
        self, policy: Union[torch.Tensor, np.ndarray, Iterable[float]], *args, **kwargs
    ):
        """
        Select action by sampling according to policy by default.
        """
        return self.rng.choice(self.action_map.actions, p=policy)

    def _select_action_eps_greedy(
        self,
        policy: Union[torch.Tensor, np.ndarray, Iterable[float]],
        eps: float,
        *args,
        **kwargs,
    ):
        """
        Select action by sampling with the argmax action having p = 1 - eps, and the remaining actions sharing the
        residual prob. mass.
        """
        argmax = int(torch.argmax(torch.tensor(policy)))
        prob = np.ones(self.action_dim) * (eps / (self.action_dim - 1))
        prob[argmax] = 1 - eps
        return self.rng.choice(self.action_map.actions, p=prob)

    def _select_action_uniform(
        self, policy: Union[torch.Tensor, np.ndarray, Iterable[float]], *args, **kwargs
    ):
        """
        Sample an action by sampling according to the policy's possible actions (p > 0).
        """
        prob = np.ones(self.action_dim) * torch.ceil(policy)
        prob = prob / prob.sum()
        return self.rng.choice(self.action_map.actions, p=prob)

    def add_reward(self, reward_token: RewardToken):
        self.reward += self.reward_map[reward_token]


class DRLAgent(RLAgent, ABC):
    """
    Deep Reinforcement Learning agent
    """

    def __init__(
        self,
        team: Union[int, Team],
        action_map: ActionMap,
        model: torch.nn.Module,
        representation: Representation,
        reward_map: Dict[RewardToken, float],
        device: str = "cpu"
    ):
        super().__init__(
            team=team,
            action_map=action_map,
            representation=representation,
            reward_map=reward_map
        )
        self.model = model
        self.device = device

    def state_to_tensor(
        self, state: State, perspective: Optional[Team] = None
    ) -> torch.Tensor:
        """
        Converts the state to a torch tensor according to the chosen representation.

        Parameters
        ----------
        state:  State,
            the state to convert.
        perspective: Team (optional),
            the team from whose perspective the representation is supposed to be built.
            Should defaults to the agent's own team, if not provided.

        Returns
        -------
        torch.Tensor,
            the state representation as tensor.
        """
        if perspective is None:
            perspective = self.team
        return self.representation.state_to_tensor(state, perspective)


class MCAgent(Agent, ABC):
    """
    A Monte Carlo Agent, which samples a consistent enemy board setup,
    given the current information and then plans according to the subclasses'
    strategy.
    """

    def __init__(self, team: Union[int, Team]):
        super().__init__(team=team)
        self.info_board = None
        self.specs = None
        self.identified_pieces: List[Piece] = []

    def setup(self, info_board: InfoBoard):
        self.info_board = info_board
        self.specs = GameSpecification(self.info_board.shape[0])

    def identify_piece(self, piece):
        """
        Update the information on the true piece type, after a fight occurred.

        Parameters
        ----------
        piece: Piece,
            the piece on which information has been gained.

        Returns
        -------

        """
        self.info_board[piece.position] = piece
        self.identified_pieces.append(piece)

    def update_info_by_move(self, move: Move, moving_piece: ShadowPiece):
        """
        update the information about the given piece, after it did the given move

        Parameters
        ----------
        move: Move,
            the move the piece made
        moving_piece: Piece,
            the piece that moved
        """
        move_dist = len(move)
        if move_dist > 1:
            new_version = sum(1 for piece in self.identified_pieces if piece.token == 2)
            self.identify_piece(
                Piece(
                    moving_piece.position,
                    moving_piece.team,
                    Token(2),
                    version=new_version,
                    hidden=False,
                )
            )
        else:
            moving_piece.exclude(Token.flag, Token.bomb)

    def draw_board_from_info(self):
        """
        Draw a setup of the enemies pieces on the board provided that aligns with the current status of
        information about said pieces, then place them on the board. This is done via iterative random sampling,
        until a consistent draw occurs. This draw may or may not represent the overall true distribution of the pieces.
        """
        # get information about enemy pieces (how many, which alive, which types, and indices in assign. array)
        board = Board(copy.deepcopy(self.info_board))
        tokens = copy.deepcopy(self.specs.token_count)
        version_table = defaultdict(list)
        for piece in self.identified_pieces:
            tokens[piece.token] -= 1
            version_table[piece.token].append(piece.version)

        enemy_pieces_to_assign = [
            piece
            for piece in board.flatten()
            if piece.team == self.team.opponent() and piece.hidden
        ]
        # place this draw now on the board by assigning the tokens
        for piece in enemy_pieces_to_assign:
            drawn_token = self.rng.choice(list(tokens.keys()))
            if tokens[drawn_token] == 1:
                tokens.pop(drawn_token)
            version = 0
            while version in version_table[drawn_token]:
                version += 1
            board[piece.position] = Piece(
                piece.position, piece.team, drawn_token, version=version
            )
        return board

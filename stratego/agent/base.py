from typing import Dict, List, Callable, Optional, Union, Iterable

import numpy as np
import copy

# from collections import Counter
from scipy import spatial

# from scipy import optimize

import torch
from collections import defaultdict
from abc import ABC

from stratego.engine import (
    State,
    Logic,
    Team,
    HookPoint,
    GameSpecification,
    Board,
    Move,
    Position,
    Piece,
    ActionMap,
    Action,
)
from stratego.learning import RewardToken, Representation, DefaultRepresentation, PolicyMode
import stratego.utils as utils


class Agent(ABC):
    """
    A general abstract agent base class.
    """

    def __init__(self, team: Union[int, Team]):
        self.team = Team(team)
        self.hooks: Dict[HookPoint, List[Callable]] = defaultdict(list)

    def decide_move(self, state: State, logic: Logic = Logic()) -> Move:
        """
        Decide the move to make for the given state of the game.

        Parameters
        ----------

        state: State,
            the state on which the decision is to be made.
        logic: Logic,
            the logic to use in the engine. Can be changed to vary the game mode if desirable.

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
        model: torch.nn.Module,
        representation: Representation,
        reward_map: Dict[RewardToken, float],
        device: str = "cpu",
        seed: Union[int, np.random.RandomState, np.random.Generator] = None,
    ):
        super().__init__(team=team)
        self.action_map = action_map
        self.action_dim = len(action_map.actions)
        self.model = model
        self.representation = representation
        self.device = device
        self.rng = utils.rng_from_seed(seed)

        self.reward = 0
        self.reward_map: Dict[RewardToken, float] = reward_map

    def sample_action(
        self,
        policy: Union[torch.Tensor, np.ndarray, Iterable[float]],
        *args,
        mode: PolicyMode,
        **kwargs
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
            raise ValueError(f"mode {mode} not supported. Allowed values are {[e.value for e in PolicyMode]}.")

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
        **kwargs
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
        self,
        policy: Union[torch.Tensor, np.ndarray, Iterable[float]],
        *args,
        **kwargs
    ):
        """
        Sample an action by sampling according to the policy's possible actions (p > 0).
        """
        prob = np.ones(self.action_dim) * torch.ceil(policy)
        prob = prob / prob.sum()
        return self.rng.choice(self.action_map.actions, p=prob)

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

    def add_reward(self, reward_token: RewardToken):
        self.reward += self.reward_map[reward_token]


class MCAgent(Agent, ABC):
    """
    A Monte Carlo Agent, which samples a consistent enemy board setup,
    given the current information and then plans according to the subclasses'
    strategy.
    """

    def __init__(self, team: Union[int, Team], game_size: int):
        super().__init__(team=team)
        self.knowledge_board: Dict[Position, Piece] = self._construct_knowledge_board(
            GameSpecification(game_size).token_count
        )
        self.identified_pieces: List[Piece]

    def _construct_knowledge_board(self, board: Board):
        kb = dict()
        return kb

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
        piece.potential_types = [enemy_int]

    def update_prob_by_move(self, move, moving_piece):
        """
        update the information about the given piece, after it did the given move
        :param move: tuple of positions tuples
        :param moving_piece: object of class Piece
        :return: change is in-place, no value specified
        """
        move_dist = spatial.distance.cityblock(move[0], move[1])
        if move_dist > 1:
            moving_piece.hidden = False
            moving_piece.potential_types = [moving_int]  # piece is 2
        else:
            immobile_enemy_types = [
                idx
                for idx, type_ in enumerate(moving_piece.potential_types)
                if type_ in [0, 11]
            ]
            moving_piece.potential_types = np.delete(
                moving_piece.potential_types, immobile_enemy_types
            )

    def draw_consistent_enemy_setup(self, board):
        """
        Draw a setup of the enemies pieces on the board provided that aligns with the current status of
        information about said pieces, then place them on the board. This is done via iterative random sampling,
        until a consistent draw occurs. This draw may or may not represent the overall true distribution of the pieces.
        :param board: numpy array (5, 5)
        :return: board with the assigned enemy pieces in it.
        """
        # get information about enemy pieces (how many, which alive, which types, and indices in assign. array)
        enemy_pieces = copy.deepcopy(self.ordered_opp_pieces)
        enemy_pieces_alive = [piece for piece in enemy_pieces if not piece.dead]
        types_alive = [int for piece in enemy_pieces_alive]

        # do the following as long as the drawn assignment is not consistent with the current knowledge about them
        consistent = False
        sample = None
        while not consistent:
            # choose as many pieces randomly as there are enemy pieces alive
            sample = np.random.choice(types_alive, len(types_alive), replace=False)
            # while-loop break condition
            consistent = True
            for idx, piece in enumerate(enemy_pieces_alive):
                # if the drawn type doesn't fit the potential types of the current piece, then redraw
                if sample[idx] not in piece.potential_types:
                    consistent = False
                    break
        # place this draw now on the board by assigning the types and changing critical attributes
        for idx, piece in enumerate(enemy_pieces_alive):
            # add attribute of the piece being guessed (only happens in non-real gameplay aka planning)
            piece.guessed = not piece.hidden
            int = sample[idx]
            if int in [0, 11]:
                piece.can_move = False
                piece.move_range = 0
            elif int == 2:
                piece.can_move = True
                piece.move_range = float("inf")
            else:
                piece.can_move = True
                piece.move_range = 1
            piece.hidden = False
            board[piece.position] = piece
        return board

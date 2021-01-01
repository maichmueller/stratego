from typing import Dict, List, Callable

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
)
from stratego.engine.action import ActionMap, Action
from stratego.learning import RewardToken


class Agent(ABC):
    """
    A general abstract agent base class.
    """

    def __init__(self, team: Team):
        self.team = team
        self.hooks: Dict[HookPoint, List[Callable]] = defaultdict(list)

    def decide_move(self, state: State, logic: Logic = Logic()) -> Move:
        """
        Decide the move to make for the given state of the engine.

        Parameters
        ----------

        state: State,
            the state on which the decision is to be made.
        logic: Logic,
            the logic to use in the engine. Can be changed to vary the engine mode if desirable.

        Returns
        -------
        Move,
            the chosen move to make on the state.
        """
        raise NotImplementedError

    def _register_hook(self, hook_point: HookPoint, hook: Callable):
        """
        Add a hook which is supposed to be provided to the game.
        """
        self.hooks[hook_point].append(hook)


class RLAgent(Agent, ABC):
    """
    Reinforcement Learning agent
    """

    def __init__(
        self,
        team: Team,
        action_map: ActionMap,
        model: torch.nn.Module,
        reward_map: Dict[RewardToken, float],
    ):
        super().__init__(team=team)
        self.action_map = action_map
        self.action_dim = len(action_map.actions)
        self.model = model
        self.total_reward = 0

        # RL attributes
        self.score = 0
        self.total_reward = 0

        self.reward_map: Dict[RewardToken, float] = reward_map

    def choose_action(self, state) -> Action:
        """
        Choose the action to take with which to form the next move.

        Parameters
        ----------
        state: State,
            the current state of the game.

        Returns
        -------
        Action,
            the chosen action.
        """
        raise NotImplementedError

    def state_to_tensor(self, state: State) -> torch.Tensor:
        """
        Converts the state to a torch tensor according to the chosen representation.

        Parameters
        ----------
        state:  State,
            the state to convert.

        Returns
        -------
        torch.Tensor
        """
        raise NotImplementedError

    def add_reward(self, reward_token: RewardToken):
        self.total_reward += self.reward_map[reward_token]


class MCAgent(Agent, ABC):
    """
    A Monte Carlo Agent, which samples a consistent enemy board setup,
    given the current information and then plans according to the subclasses'
    strategy.
    """

    def __init__(self, team: Team, game_size: int):
        super().__init__(team)
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

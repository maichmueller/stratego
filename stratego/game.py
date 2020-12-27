import copy

from .piece import Piece
from copy import deepcopy
from typing import Tuple, Optional, Dict, List, Sequence

import numpy as np

from .utils import *
from collections import defaultdict
from .agent import Agent, RLAgent
from .state import State
from .logic import Logic
from .spatial import Position, Board, Move
from .game_defs import Status, Player

from inspect import signature
import matplotlib.pyplot as plt


def get_game_defaults(board_size: str):
    board_size = board_size.lower()
    if board_size in ["s", "small"]:
        types_available = np.array([0, 1] + [2] * 3 + [3] * 2 + [10] + [11] * 2)
        obstacle_positions = [(2, 2)]
        game_size = 5
    elif board_size in ["m", "medium"]:
        types_available = np.array(
            [0, 1] + [2] * 5 + [3] * 3 + [4] * 3 + [5] * 2 + [6] + [10] + [11] * 4
        )
        obstacle_positions = [(3, 1), (3, 5)]
        game_size = 7
    elif board_size in ["l", "large"]:
        types_available = np.array(
            [0, 1]
            + [2] * 8
            + [3] * 5
            + [4] * 4
            + [5] * 4
            + [6] * 4
            + [7] * 3
            + [8] * 2
            + [9] * 1
            + [10]
            + [11] * 6
        )
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
        game_size = 10
    else:
        raise ValueError(f"Board size {board_size} not supported.")
    return types_available, obstacle_positions, game_size


class Game:
    def __init__(
        self,
        agent0: Agent,
        agent1: Agent,
        state: Optional[State] =None,
        board_size: str = "l",
        fixed_setups: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
    ):
        self.board_size = board_size
        self.agents = (agent0, agent1)
        self.fixed_setups = fixed_setups

        (
            self.types_available,
            self.obstacle_positions,
            self.game_size,
        ) = get_game_defaults(board_size)

        self.history = None
        self.state = state
        if state is not None:
            Logic.compute_dead_pieces(state.board, self.types_available)
        else:
            self.reset()

        # reinforcement learning attributes
        self.score = 0
        self.reward = 0
        self.steps = 0
        self.death_steps = None
        self.illegal_moves = 0

        self.reward_illegal = 0  # punish illegal moves
        self.reward_step = 0  # negative reward per agent step
        self.reward_win = 1  # win game
        self.reward_loss = -1  # lose game
        self.reward_kill = 0  # kill enemy figure reward
        self.reward_die = 0  # lose to enemy figure

    def __str__(self):
        return np.array_repr(self.state.board)

    def __hash__(self):
        return hash(str(self))

    def _build_board_from_setups(self, setup0, setup1):
        board = np.empty((self.game_size, self.game_size), dtype=object)

        for setup in (setup0, setup1):
            pieces_version = defaultdict(int)
            for idx, piece in np.ndenumerate(setup):
                if piece is not None:
                    pieces_version[int] += 1
                    board[piece.position] = deepcopy(piece)

        for pos in self.obstacle_positions:
            obs = Piece(99, 99, Position(pos[0], pos[1]))
            obs.hidden = False
            board[pos] = obs

        return board

    def reset(self):
        setup0 = self.fixed_setups[0]
        if setup0 is None:
            setup0 = self._draw_random_setup(self.types_available, 0, self.game_size)
        setup1 = self.fixed_setups[1]
        if setup1 is None:
            setup1 = self._draw_random_setup(self.types_available, 1, self.game_size)

        board_arr = self._build_board_from_setups(setup0, setup1)

        self.state = State(Board(board_arr))

        self.history = History()

        return self

    def run_game(self, show=False, **kwargs):
        game_over = False
        rewards = None
        block = kwargs.pop("block", False)
        kwargs_print = {
            p.name: kwargs[p.name]
            for p in signature(Board.print_board).parameters.values()
            if p in kwargs
        }
        kwargs_run_step = {
            p.name: kwargs[p.name]
            for p in signature(self.run_step).parameters.values()
            if p in kwargs
        }
        if show:

            def print_board():
                self.state.board.print_board(**kwargs_print)
                plt.show(block=block)

        else:

            def print_board():
                pass

        if (status := Logic.get_status(self.state)) != Status.ongoing:
            rewards = status
            game_over = True

        while not game_over:
            print_board()
            rewards = self.run_step(**kwargs_run_step)
            if rewards != Status.ongoing:
                game_over = True
        print_board()
        return rewards

    def run_step(self, move: Optional[Move] = None, **kwargs):
        player = self.state.active_player
        agent = self.agents[player]

        if move is None:
            move = agent.decide_move(self.state, **kwargs)

        # test if agent can't move anymore
        if move is None:
            return Status.draw

        if not Logic.is_legal_move(self.state.board, move):
            self.reward_agent(agent, self.reward_illegal)
            return Status.win_red if player == Player.Team.blue else Status.win_blue

        self.history.commit_move(
            self.state.board,
            move,
            player,
        )

        fight_status = Logic.execute_move(self.state, move)  # execute agent's choice

        if fight_status is not None:
            if fight_status == 1:
                self.reward_agent(agent, self.reward_kill)
            elif fight_status == -1:
                self.reward_agent(agent, -self.reward_kill)

        # test if game is over
        if Logic.get_status(self.state) != Status.ongoing:
            return self.state.terminal

        self.state.move_counter += 1

        return Status.ongoing

    @staticmethod
    def _draw_random_setup(types_available: Sequence[int], player: Player, game_size: int):
        """
        Draw a random setup from the set of types types_available after placing the flag
        somewhere in the last row of the board of the side of 'team', or behind the obstacle.
        
        Parameters
        ----------
        types_available:    list,
            piece types to draw from
        player: Player,
        game_size: int,
            the board size
        
        Returns
        -------
        np.ndarray,
            the setup, in numpy array form
        """
        nr_pieces = len(types_available) - 1
        types_available = [type_ for type_ in types_available if not type_ == 0]
        if game_size == 5:
            row_offset = 2
        elif game_size == 7:
            row_offset = 3
        else:
            row_offset = 4
        setup_agent = np.empty((row_offset, game_size), dtype=object)
        if player == 0:
            flag_positions = [(game_size - 1, j) for j in range(game_size)]
            flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
            flag_pos = flag_positions[flag_choice]
            flag_pos_inv = (
                game_size - 1 - flag_positions[flag_choice][0],
                game_size - 1 - flag_positions[flag_choice][1],
            )
            setup_agent[flag_pos_inv] = Piece(0, 0, Position(flag_pos[0], flag_pos[1]))

            types_draw = np.random.choice(types_available, nr_pieces, replace=False)
            positions_agent_0 = [
                (i, j)
                for i in range(game_size - row_offset, game_size)
                for j in range(game_size)
            ]
            positions_agent_0.remove(flag_positions[flag_choice])

            for idx in range(nr_pieces):
                pos = positions_agent_0[idx]
                setup_agent[(game_size - 1 - pos[0], game_size - 1 - pos[1])] = Piece(
                    types_draw[idx], 0, Position(pos[0], pos[1])
                )
        elif player == 1:
            flag_positions = [(0, j) for j in range(game_size)]
            flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
            flag_pos = flag_positions[flag_choice]
            flag_pos_inv = (
                game_size - 1 - flag_positions[flag_choice][0],
                game_size - 1 - flag_positions[flag_choice][1],
            )
            setup_agent[flag_pos_inv] = Piece(0, 1, Position(flag_pos[0], flag_pos[1]))

            types_draw = np.random.choice(types_available, nr_pieces, replace=False)
            positions_agent_1 = [
                (i, j) for i in range(row_offset) for j in range(game_size)
            ]
            positions_agent_1.remove(flag_positions[flag_choice])

            for idx in range(nr_pieces):
                pos = positions_agent_1[idx]
                setup_agent[pos] = Piece(types_draw[idx], 1, Position(pos[0], pos[1]))
        return setup_agent

    @staticmethod
    def reward_agent(agent: RLAgent, reward: float):
        if isinstance(agent, RLAgent):
            agent.add_reward(reward)


class History:

    def __init__(self):
        self.turns: List[int] = []
        self.move: Dict[int, Move] = dict()
        self.team: Dict[int, int] = dict()
        self.pieces: Dict[int, Tuple[Piece, Piece]] = dict()

    def undo_last_n_turns(self, n: int, board: Board):
        """
        Undo the last n moves in the memory.
        board:  Board,
            the state on which to undo the last rounds
        n:  int,
            number of moves to undo
        """
        for t in range(n):
            turn = self.turns[-1] - t
            move, (piece_from, piece_to) = self.move[turn], self.pieces[turn]
            board[move.from_] = piece_from
            board[move.to_] = piece_to
            piece_from.change_position(move.from_)
            piece_to.change_position(move.to_)

    def commit_move(self, board: Board, move: Move, turn: int):
        """
        Commit the current move to history.
        """
        from_ = move.from_
        to_ = move.to_
        self.move[turn] = move
        self.pieces[turn] = copy.deepcopy(board[from_]), copy.deepcopy(board[to_])
        self.team[turn] = turn % 2





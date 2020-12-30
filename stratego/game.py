from .learning import RewardToken
from .piece import Piece
from .agent.base import Agent, RLAgent
from .state import State
from .logic import Logic
from .spatial import Position, Move, Board
from .game_defs import Status, Player, get_game_specs, Team, HookPoint
from .utils import slice_kwargs

from copy import deepcopy
from typing import Tuple, Optional, Dict, List, Sequence, Callable, Iterable
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt


class Game:
    def __init__(
        self,
        agent0: Agent,
        agent1: Agent,
        state: Optional[State] = None,
        game_size: str = "l",
        fixed_setups: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
    ):
        self.game_size = game_size
        self.agents: Dict[Player, Agent] = {
            Player(agent0.team): agent0,
            Player(agent1.team): agent1,
        }
        self.hook_handler: Dict[HookPoint, List[Callable]] = defaultdict(list)
        self._gather_hooks(agents=(agent0, agent1))
        self.fixed_setups = fixed_setups

        (self.token_count, self.obstacle_positions, self.game_size) = get_game_specs(
            game_size
        )

        self.state: State
        if state is not None:
            self.state = state
            Logic.compute_dead_pieces(state.board, self.token_count)
        else:
            self.reset()

    def __str__(self):
        return np.array_repr(self.state.board)

    def __hash__(self):
        return hash(str(self))

    def _gather_hooks(self, agents: Iterable[Agent]):
        for agent in agents:
            for hook_point, hooks in agent.hooks.items():
                self.hook_handler[hook_point].extend(hooks)

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
            setup0 = self._draw_random_setup(
                self.token_count, Player(0), self.game_size
            )
        setup1 = self.fixed_setups[1]
        if setup1 is None:
            setup1 = self._draw_random_setup(
                self.token_count, Player(1), self.game_size
            )

        board_arr = self._build_board_from_setups(setup0, setup1)

        self.state = State(Board(board_arr))

        return self

    def run_game(self, show=False, **kwargs):
        game_over = False
        block = kwargs.pop("block", False)
        kwargs_print = slice_kwargs(Board.print_board, kwargs)
        kwargs_run_step = slice_kwargs(self.run_step, kwargs)
        if show:
            # if the game progress should be shown, then we refer to the board print method.
            def print_board():
                self.state.board.print_board(**kwargs_print)
                plt.show(block=block)

        else:
            # if the game progress should not be shown, then we simply pass over this step.
            def print_board():
                pass

        if (status := Logic.get_status(self.state)) != Status.ongoing:
            game_over = True

        self._trigger_hooks(HookPoint.pre_run, self.state)

        while not game_over:
            print_board()
            status = self.run_step(**kwargs_run_step)
            if status != Status.ongoing:
                game_over = True
        print_board()

        self._trigger_hooks(HookPoint.post_run, self.state, status)

        return status

    def run_step(self, move: Optional[Move] = None):
        """
        Execute one step of the game (i.e. the action decided by the active player).

        Parameters
        ----------
        move: Move (optional),
            hijack parameter, if the move should be decided from outside and not the active agent itself.

        Returns
        -------
        Status,
            the current status of the game.
        """
        player = self.state.active_player
        agent = self.agents[player]

        self._trigger_hooks(HookPoint.pre_move_decision, self.state)

        if move is None:
            move = agent.decide_move(self.state.get_info_state(player))

        self._trigger_hooks(HookPoint.post_move_decision, self.state, move)

        if not Logic.is_legal_move(self.state.board, move):
            self.reward_agent(agent, RewardToken.illegal)
            return Status.win_red if player.team == Team.blue else Status.win_blue

        self.state.history.commit_move(
            self.state.board,
            move,
            player,
        )

        self._trigger_hooks(HookPoint.pre_move_execution, self.state, move)

        fight_status = Logic.execute_move(self.state, move)  # execute agent's choice

        self._trigger_hooks(
            HookPoint.post_move_execution, self.state, move, fight_status
        )

        if fight_status is not None:
            if fight_status == 1:
                self.reward_agent(agent, RewardToken.kill)
            elif fight_status == -1:
                self.reward_agent(agent, RewardToken.die)
            else:
                self.reward_agent(agent, RewardToken.kill_mutually)

        # test if game is over
        if (status := Logic.get_status(self.state)) != Status.ongoing:
            return status

        self.state.move_counter += 1

        return Status.ongoing

    def _trigger_hooks(self, hook_point: HookPoint, *args, **kwargs):
        for hook in self.hook_handler[hook_point]:
            hook(*args, **kwargs)

    @staticmethod
    def _draw_random_setup(token_count: Sequence[int], player: Player, game_size: int):
        """
        Draw a random setup from the set of types token_count after placing the flag
        somewhere in the last row of the board of the side of 'team', or behind the obstacle.

        Parameters
        ----------
        token_count:    list,
            piece types to draw from
        player: Player,
        game_size: int,
            the board size

        Returns
        -------
        np.ndarray,
            the setup, in numpy array form
        """
        nr_pieces = len(token_count) - 1
        token_count = [type_ for type_ in token_count if not type_ == 0]
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

            types_draw = np.random.choice(token_count, nr_pieces, replace=False)
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

            types_draw = np.random.choice(token_count, nr_pieces, replace=False)
            positions_agent_1 = [
                (i, j) for i in range(row_offset) for j in range(game_size)
            ]
            positions_agent_1.remove(flag_positions[flag_choice])

            for idx in range(nr_pieces):
                pos = positions_agent_1[idx]
                setup_agent[pos] = Piece(types_draw[idx], 1, Position(pos[0], pos[1]))
        return setup_agent

    @staticmethod
    def reward_agent(agent: Agent, reward: RewardToken):
        if isinstance(agent, RLAgent):
            agent.add_reward(reward)

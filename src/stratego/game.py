from stratego.learning import default_reward_function, RewardToken
from stratego.agent import Agent, RLAgent
from stratego.utils import slice_kwargs
from stratego.core import (
    State,
    InfoState,
    Logic,
    Position,
    Move,
    Board,
    Status,
    Team,
    HookPoint,
    GameSpecification,
    Piece,
    Obstacle,
)

import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt

from typing import Optional, Dict, List, Sequence, Callable, Iterable, Union, Tuple


class Game:
    def __init__(
        self,
        agent0: Agent,
        agent1: Agent,
        state: Optional[State] = None,
        game_size: str = "l",
        logic: Logic = Logic(),
        reward_func: Callable[[RewardToken], float] = default_reward_function,
        fixed_setups: Tuple[Optional[Iterable[Piece]]] = (None, None),
        seed: Optional[Union[np.random.Generator, int]] = None,
    ):
        self.agents: Dict[Team, Agent] = {
            Team(agent0.team): agent0,
            Team(agent1.team): agent1,
        }
        self.hook_handler: Dict[HookPoint, List[Callable]] = defaultdict(list)
        self._register_hooks(agents=(agent0, agent1))
        self.fixed_setups: Dict[Team, Optional[Sequence[Piece]]] = dict()
        for team in Team:
            if setup := fixed_setups[team.value] is not None:
                self.fixed_setups[team] = tuple(setup)
            else:
                self.fixed_setups[team] = None

        self.specs: GameSpecification = GameSpecification(game_size)

        self.rng_state = np.random.default_rng(seed)
        self.logic = logic
        self.state: State
        if state is not None:
            self.state = state
            self.state.dead_pieces = logic.compute_dead_pieces(
                state.board, self.specs.token_count
            )
        else:
            self.reset()
        self.reward_func = reward_func

    def __str__(self):
        return (
            f"Agent Blue: {self.agents[Team.blue]}\n"
            f"Agent Red:  {self.agents[Team.red]}\n"
            f"Game size: {self.specs.game_size}\n"
            f"Logic: {type(self.logic).__name__}\n"
            f"State:\n{str(self.state)}"
        )

    def __hash__(self):
        return hash(str(self))

    def _register_hooks(self, agents: Iterable[Agent]):
        for agent in agents:
            for hook_point, hooks in agent.hooks.items():
                self.hook_handler[hook_point].extend(hooks)

    def reset(self):
        self.state = State(
            Board(self.draw_board()),
            starting_team=self.rng_state.choice([Team.blue, Team.red]),
        )
        return self

    def run_game(self, show: bool = False, pause: float = 0.0, **kwargs):
        game_over = False
        block = kwargs.pop("block", False)
        kwargs_print = slice_kwargs(Board.print_board, kwargs)
        kwargs_run_step = slice_kwargs(self.run_step, kwargs)
        if show:
            # if the core progress should be shown, then we refer to the board print method.
            def print_board():
                self.state.board.print_board(**kwargs_print)
                plt.pause(pause)
                plt.show(block=block)

        else:
            # if the core progress should not be shown, then we simply pass over this step.
            def print_board():
                pass

        if (
            status := self.logic.get_status(self.state, specs=self.specs)
        ) != Status.ongoing:
            game_over = True

        self._trigger_hooks(HookPoint.pre_run, self.state)

        while not game_over:
            print_board()
            status = self.run_step(**kwargs_run_step)
            if status != Status.ongoing:
                game_over = True

        if status == Status.win_blue:
            self.reward_agent(
                self.agents[Team.blue], default_reward_function(RewardToken.win)
            )
            self.reward_agent(
                self.agents[Team.red], default_reward_function(RewardToken.loss)
            )
        elif status == Status.win_red:
            self.reward_agent(
                self.agents[Team.blue], default_reward_function(RewardToken.loss)
            )
            self.reward_agent(
                self.agents[Team.red], default_reward_function(RewardToken.win)
            )
        else:
            # it's a tie. No rewards.
            pass

        print_board()

        self._trigger_hooks(HookPoint.post_run, self.state, status)

        return status

    def run_step(self, move: Optional[Move] = None) -> Status:
        """
        Execute one step of the core (i.e. the action decided by the active player).

        Parameters
        ----------
        move: Move (optional),
            hijack parameter, if the move should be decided from outside and not the active agent itself.

        Returns
        -------
        Status,
            the current status of the core.
        """
        player = self.state.active_team
        agent = self.agents[player]

        self._trigger_hooks(HookPoint.pre_move_decision, self.state)

        if move is None:
            move = agent.decide_move(InfoState(self.state, player), logic=self.logic)

        self._trigger_hooks(HookPoint.post_move_decision, self.state, move)

        if not self.logic.is_legal_move(self.state.board, move):
            self.reward_agent(agent, self.reward_func(RewardToken.illegal))
            return Status.win_red if player == Team.blue else Status.win_blue

        self.state.history.commit_move(
            self.state.board, move, self.state.turn_counter,
        )

        self._trigger_hooks(HookPoint.pre_move_execution, self.state, move)

        fight_status = self.logic.execute_move(
            self.state, move
        )  # execute agent's choice

        self._trigger_hooks(
            HookPoint.post_move_execution, self.state, move, fight_status
        )

        if fight_status is not None:
            if fight_status == 1:
                self.reward_agent(agent, self.reward_func(RewardToken.kill))
            elif fight_status == -1:
                self.reward_agent(agent, self.reward_func(RewardToken.die))
            else:
                self.reward_agent(agent, self.reward_func(RewardToken.kill_mutually))

        # test if game is over
        if (
            status := self.logic.get_status(self.state, specs=self.specs)
        ) != Status.ongoing:
            return status

        return Status.ongoing

    def _trigger_hooks(self, hook_point: HookPoint, *args, **kwargs):
        for hook in self.hook_handler[hook_point]:
            hook(*args, **kwargs)

    def draw_board(self):
        """
        Draw a random board according to the current core specification.

        Returns
        -------
        np.ndarray,
            the setup, in numpy array form
        """
        rng = self.rng_state

        board = Board(
            np.empty((self.specs.game_size, self.specs.game_size), dtype=object)
        )  # inits all entries to None
        for team in Team:
            token_count = self.specs.token_count
            all_tokens = list(token_count.keys())
            token_freqs = list(token_count.values())

            if (setup := self.fixed_setups[team]) is not None:
                for piece in setup:
                    board[piece.position] = piece
            else:
                setup_rows = self.specs.setup_rows[team]

                all_pos = [
                    Position(r, c)
                    for r, c in itertools.product(
                        setup_rows, range(self.specs.game_size)
                    )
                ]

                while all_pos:

                    token_draw = rng.choice(
                        np.arange(len(all_tokens)),
                        p=list(map(lambda x: x / sum(token_freqs), token_freqs)),
                    )
                    token = all_tokens[token_draw]
                    version = token_freqs[token_draw]
                    token_freqs[token_draw] -= 1
                    if token_freqs[token_draw] == 0:
                        # if no such token is left to be drawn, then remove it from the token list
                        all_tokens.pop(token_draw)
                        token_freqs.pop(token_draw)

                    pos_draw = rng.choice(np.arange(len(all_pos)))
                    pos = all_pos[pos_draw]
                    all_pos.pop(pos_draw)

                    board[pos] = Piece(pos, team, token, version)

        for obs_pos in self.specs.obstacle_positions:
            board[obs_pos] = Obstacle(obs_pos)

        return board

    @staticmethod
    def reward_agent(agent: Agent, reward: float):
        if isinstance(agent, RLAgent):
            agent.add_reward(reward)

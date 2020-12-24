from copy import deepcopy
from typing import Tuple, Optional

import numpy as np

import utils
import pieces
from collections import Counter, defaultdict
from agent import Agent, RLAgent
from state import State


def get_game_defaults(board_size: str):
    board_size = board_size.lower()
    if board_size in ["s", "small"]:
        types_available = np.array([0, 1] + [2] * 3 + [3] * 2 + [10] + [11] * 2)
        obstacle_positions = [(2, 2)]
        game_dim = 5
    elif board_size in ["m", "medium"]:
        types_available = np.array(
            [0, 1] + [2] * 5 + [3] * 3 + [4] * 3 + [5] * 2 + [6] + [10] + [11] * 4
        )
        obstacle_positions = [(3, 1), (3, 5)]
        game_dim = 7
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
        game_dim = 10
    else:
        raise ValueError(f"Board size {board_size} not supported.")
    return types_available, obstacle_positions, game_dim


class Game:
    def __init__(
        self,
        agent0: Agent,
        agent1: Agent,
        board_size: str = "l",
        fixed_setups: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
    ):
        self.board_size = board_size
        self.agents = (agent0, agent1)
        self.fixed_setups = fixed_setups

        (
            self.types_available,
            self.obstacle_positions,
            self.game_dim,
        ) = get_game_defaults(board_size)

        self.state = None
        self.replay = None
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
        board = np.empty((self.game_dim, self.game_dim), dtype=object)

        for setup in (setup0, setup1):
            pieces_version = defaultdict(int)
            for idx, piece in np.ndenumerate(setup):
                if piece is not None:
                    pieces_version[piece.type] += 1
                    piece.version = pieces_version[piece.type]
                    board[piece.position] = deepcopy(piece)

        for pos in self.obstacle_positions:
            obs = pieces.Piece(99, 99, pos)
            obs.hidden = False
            board[pos] = obs

        return board

    def reset(self):
        if self.fixed_setups[0] is None:
            self.agents[0].setup = self._draw_random_setup(
                self.types_available, 0, self.game_dim
            )
        else:
            self.agents[0].setup = self.fixed_setups[0]
        if self.fixed_setups[1] is None:
            self.agents[1].setup = self._draw_random_setup(
                self.types_available, 1, self.game_dim
            )
        else:
            self.agents[1].setup = self.fixed_setups[1]

        if self.agents[0].setup is not None and self.agents[1].setup is not None:
            board = self._build_board_from_setups(
                self.agents[0].setup, self.agents[1].setup
            )
        else:
            raise ValueError("Missing board information.")

        self.state = State(board)
        self.agents[0].install_board(self.state.board, reset=True)
        self.agents[1].install_board(self.state.board, reset=True)

        action_rep = utils.action_rep

        if isinstance(self.agents[0], RLAgent):
            self.agents[0].set_action_rep(
                actors=action_rep.actors,
                actions=action_rep.actions,
                relation_dict=action_rep.piecetype_to_actionrange,
            )
        if isinstance(self.agents[1], RLAgent):
            self.agents[1].set_action_rep(
                actors=action_rep.actors,
                actions=action_rep.actions,
                relation_dict=action_rep.piecetype_to_actionrange,
            )

        self.replay = GameReplay(self.state.board)

        self.move_count = 1  # agent 1 starts

        return self

    def run_game(self, show=False, **kwargs):
        game_over = False
        rewards = None
        if show:
            print_board = utils.print_board
        else:

            def print_board(*unused, **unusedkw):
                pass

        while not game_over:
            print_board(self.state.board)
            rewards = self.run_step(**kwargs)
            if rewards != 404:
                game_over = True
        print_board(self.state.board)
        return rewards

    def run_step(self, move=None, **kwargs):
        turn = self.move_count % 2  # player 1 or player 0
        # print(self.state.board)
        if move is None:
            move = self.agents[turn].decide_move(**kwargs)

        # test if agent can't move anymore
        if move is None:
            if turn == 1:
                return 2  # agent0 wins
            else:
                return -2  # agent1 wins

        if not utils.is_legal_move(self.state.board, move):
            self.agents[turn].decide_move(**kwargs)

        # let agents update their boards
        for agent in self.agents:
            agent.do_move(move, true_gameplay=True)

        self.replay.add_move(
            move,
            (self.state.board[move[0]], self.state.board[move[1]]),
            turn,
            self.move_count,
        )
        outcome = self.state.do_move(move)  # execute agent's choice

        if outcome is not None:
            self._update_fight_rewards(outcome, turn)

        # test if game is over
        terminal = self.state.is_terminal(flag_only=True)
        if terminal != 404:  # flag discovered, or draw
            if terminal > 0:
                x = 3
            return terminal

        self.move_count += 1
        for agent_ in self.agents:
            agent_.move_count = self.move_count
        return 404

    @staticmethod
    def _draw_random_setup(types_available, team, game_dim):
        """
        Draw a random setup from the set of types types_available after placing the flag
        somewhere in the last row of the board of the side of 'team', or behind the obstacle.
        :param types_available: list of types to draw from, integers
        :param team: boolean, 1 or 0 depending on the team
        :param game_dim: int, the board dimension
        :return: the setup, in numpy array form
        """
        nr_pieces = len(types_available) - 1
        types_available = [type_ for type_ in types_available if not type_ == 0]
        if game_dim == 5:
            row_offset = 2
        elif game_dim == 7:
            row_offset = 3
        else:
            row_offset = 4
        setup_agent = np.empty((row_offset, game_dim), dtype=object)
        if team == 0:
            flag_positions = [(game_dim - 1, j) for j in range(game_dim)]
            flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
            flag_pos = (
                game_dim - 1 - flag_positions[flag_choice][0],
                game_dim - 1 - flag_positions[flag_choice][1],
            )
            setup_agent[flag_pos] = pieces.Piece(0, 0, flag_positions[flag_choice])

            types_draw = np.random.choice(types_available, nr_pieces, replace=False)
            positions_agent_0 = [
                (i, j)
                for i in range(game_dim - row_offset, game_dim)
                for j in range(game_dim)
            ]
            positions_agent_0.remove(flag_positions[flag_choice])

            for idx in range(nr_pieces):
                pos = positions_agent_0[idx]
                setup_agent[
                    (game_dim - 1 - pos[0], game_dim - 1 - pos[1])
                ] = pieces.Piece(types_draw[idx], 0, pos)
        elif team == 1:
            flag_positions = [(0, j) for j in range(game_dim)]
            flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
            setup_agent[flag_positions[flag_choice]] = pieces.Piece(
                0, 1, flag_positions[flag_choice]
            )

            types_draw = np.random.choice(types_available, nr_pieces, replace=False)
            positions_agent_1 = [
                (i, j) for i in range(row_offset) for j in range(game_dim)
            ]
            positions_agent_1.remove(flag_positions[flag_choice])

            for idx in range(nr_pieces):
                pos = positions_agent_1[idx]
                setup_agent[pos] = pieces.Piece(types_draw[idx], 1, pos)
        return setup_agent

    def _update_fight_rewards(self, outcome, turn):
        if outcome == 1:
            if self.agents[turn].learner:
                self.agents[turn].add_reward(self.reward_kill)
            if self.agents[(turn + 1) % 2].learner:
                self.agents[(turn + 1) % 2].add_reward(self.reward_die)
        if outcome == -1:
            if self.agents[turn].learner:
                self.agents[turn].add_reward(self.reward_die)
            if self.agents[(turn + 1) % 2].learner:
                self.agents[(turn + 1) % 2].add_reward(self.reward_kill)
        else:
            if self.agents[turn].learner:
                self.agents[turn].add_reward(self.reward_kill)
                self.agents[turn].add_reward(self.reward_die)
            if self.agents[(turn + 1) % 2].learner:
                self.agents[(turn + 1) % 2].add_reward(self.reward_kill)
                self.agents[(turn + 1) % 2].add_reward(self.reward_die)

    def _update_terminal_moves_rewards(self, turn):
        if self.agents[(turn + 1) % 2].learner:
            self.agents[(turn + 1) % 2].add_reward(self.reward_win)
        if self.agents[turn].learner:
            self.agents[turn].add_reward(self.reward_loss)

    def _update_terminal_flag_rewards(self, turn):
        if self.agents[turn].learner:
            self.agents[turn].add_reward(self.reward_win)
        if self.agents[(turn + 1) % 2].learner:
            self.agents[(turn + 1) % 2].add_reward(self.reward_loss)

    def get_action_rep(self, force=False):
        return self.state.get_action_rep(force=force)


class GameReplay:
    def __init__(self, board):
        self.initialBoard = deepcopy(board)
        self.curr_board = deepcopy(board)
        self.pieces_team_0 = []
        self.pieces_team_1 = []
        for pos, piece in np.ndenumerate(self.initialBoard):
            if piece is not None:
                if piece.team == 0:
                    self.pieces_team_0.append(piece)
                else:
                    self.pieces_team_1.append(piece)
        self.moves_and_pieces_in_round = dict()
        self.team_of_round = dict()

    def add_move(self, move, pieces, team, round):
        self.moves_and_pieces_in_round[round] = (move, pieces[0], pieces[1])
        self.team_of_round[round] = team
        self.curr_board = self.do_move(self.curr_board, move)

    def restore_to_round(self, round):
        round_dist = max(self.moves_and_pieces_in_round.keys()) - round
        board_ = self.curr_board
        if (
            round_dist > round
        ):  # deciding which way around to restore: from the beginning or the end
            # restore from end
            board_ = self.undo_last_n_moves(n=round, board=board_)
        else:
            # restore from beginning
            board_ = deepcopy(self.initialBoard)
            for played_round in range(round):
                board_ = self.do_move(
                    board_, self.moves_and_pieces_in_round[played_round][0]
                )
        return board_

    def undo_last_n_moves(self, n, board):
        """
        Undo the last n moves in the memory. Return the updated board.
        :param board: numpy array
        :param n: int number of moves to undo
        :return: board
        """
        max_round = max(self.moves_and_pieces_in_round.keys())
        for k in range(n):
            (from_, to_), piece_from, piece_to = self.moves_and_pieces_in_round[
                max_round - k
            ]
            board[from_] = piece_from
            board[to_] = piece_to
            piece_from.position = from_
            piece_to.position = to_
        return board

    def do_move(self, board, move):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        :param board: numpy array representing the board
        """
        from_ = move[0]
        to_ = move[1]
        if board[to_] is not None:  # Target field is not empty, then has to fight
            fight_outcome = Game.bm[board[from_].type, board[to_].type]
            if fight_outcome == 1:
                board[to_] = board[from_]
            elif fight_outcome == 0:
                board[to_] = None
        else:
            board[to_] = board[from_]
        board[from_] = None
        return board

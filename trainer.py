import copy as cp
from game import Game
import numpy as np
from matplotlib import pyplot as plt
from cythonize import helpers


class GameCoach(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def run_step(self):

        self.reward = 0
        self.steps += 1

        turn = self.move_count % 2  # player 1 or player 0

        new_move = self.agents[turn].decide_move()

        # test if agent can't move anymore
        if new_move is None:
            self._update_terminal_moves_rewards(turn)
            if turn == 1:
                return 2  # agent0 wins
            else:
                return -2  # agent1 wins

        for _agent in self.agents:
            _agent.do_move(new_move, true_gameplay=True)
        outcome = self.state.do_move(new_move)  # execute agent's choice

        if outcome is not None:
            self._update_fight_rewards(outcome, turn)

        # test if game is over
        terminal = self.state.is_terminal(flag_only=True, move_count=self.move_count)
        if terminal:  # flag discovered, or draw
            return terminal

        self.move_count += 1
        for agent_ in self.agents:
            agent_.move_count = self.move_count
        return 0, 0

    def _update_fight_rewards(self, outcome, turn):
        if outcome == 1:
            if self.agents[turn].learner:
                self.agents[turn].add_reward(self.reward_kill)
            if self.agents[(turn + 1) % 2].learner:
                self.agents[(turn + 1) % 2].add_reward(self.reward_die)
        if outcome == -1:
            turn += 1
            if self.agents[turn].learner:
                self.agents[turn].add_reward(self.reward_kill)
            if self.agents[(turn + 1) % 2].learner:
                self.agents[(turn + 1) % 2].add_reward(self.reward_die)
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

    def show(self):
        fig = plt.figure(1)
        helpers.print_board(self.state.board)
        plt.title("Reward = {}".format(self.score))
        fig.canvas.draw()  # updates plot

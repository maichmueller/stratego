############################################################################
#
# The whole project is written by Leonhard Bereska and Michael Aichmüller.
# It is hard, if not almost impossible to try to differentiate between the
# parts coded by Bereska and the parts by Aichmüller, keep the vague
# scheme in mind:
# The whole game code was a 50/50 split (legal_moves, do_move, utils etc.)
# meaning that while one person might have started to code a function,
# it was always revised and adapted by another, making the overall effort
# equal.
# The MiniMax part was mainly coded by Aichmüller, revision by Bereska
# The DQN learning part was mainly coded by Bereska, revision by Aichmüller.
# Yet bugfixing, and overall adaptation to the whole game flow was again
# an equal effort of both parts, thus it doesn't feel right to attribute parts
# to one person or another.
#
##############################################################################

import game
import agent
from timeit import default_timer as timer
import re

import colorama
from colorama import Fore, Style
from collections import defaultdict
colorama.init()


class Arena:
    class WinStats:
        def __init__(self):
            self.wins = 0
            self.win_reasons = defaultdict(int)
            self.rounds_count = []
            self.\
                draws = 0

        def add_win(self, reason, count=None):
            self.win_reasons[reason] += 1
            self.wins += 1
            if count is not None:
                self.rounds_count.append((1, count))

        def add_draw(self, count=None):
            self.draws += 1
            if count is not None:
                self.rounds_count.append((0, count))

        def get_count_stats(self):
            s = sum([x[0] for x in self.rounds_count])
            avg = s / len(self.rounds_count)
            return avg , s

    def __init__(self, agent_0, agent_1, game_size='small', setup_0=None, setup_1=None):
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.game_size = game_size
        self.setup_0 = setup_0
        self.setup_1 = setup_1

    def pit(self, num_sims, show_game=False, save_results=False):
        """
        Simulate num_sims many games of the agent of type agent_type_0 against the agent of
        type agent_type_1. If setup_0 or setup_1 are provided respectively, then take the pieces
        setup from those. If show_game is True, the game will be printed by the internal function.
        :param game_: game object that runs the simulation
        :param num_sims: integer number of games to simulate
        :param setup_0: (optional) numpy array of the setup of agent 0
        :param setup_1: (optional) numpy array of the setup of agent 1
        :param show_game: (optional) boolean, whether to show the game or not
        :return: None, writes results to a file named after the agents acting
        """
        blue_wins = self.WinStats()  # Agent 0
        red_wins = self.WinStats()  # Agent 1

        game_times_0 = []
        game_times_1 = []

        ag_type_0 = str(self.agent_0)
        ag_type_1 = str(self.agent_1)
        ag_type_0 = re.search('agent.(.+?) object', ag_type_0).group(1)
        ag_type_1 = re.search('agent.(.+?) object', ag_type_1).group(1)

        if self.setup_0 is None or self.setup_1 is None:
            game_ = game.Game(agent0=self.agent_0, agent1=self.agent_1, game_size=self.game_size)
        else:
            game_ = game.Game(agent0=self.agent_0, agent1=self.agent_1, game_size=self.game_size,
                              fixed_setups=(self.setup_0, self.setup_1))

        for simu in range(1, num_sims + 1):  # simulate games
            # reset setup with new setup if none given
            game_.reset()

            game_time_s = timer()

            while True:
                game_reward = game_.run_game(show_game)
                if game_reward != 404:
                    if game_reward == 1:  # 0 won by finding flag
                        game_times_0.append(timer() - game_time_s)
                        blue_wins.add_win('flag', game_.move_count)
                    elif game_reward == 2:  # 0 won by moves
                        game_times_0.append(timer() - game_time_s)
                        blue_wins.add_win('moves', game_.move_count)
                    elif game_reward == -1:  # 1 won by finding flag
                        game_times_1.append(timer() - game_time_s)
                        red_wins.add_win('flag', game_.move_count)
                    elif game_reward == -2:  # 1 won by moves
                        game_times_1.append(timer() - game_time_s)
                        red_wins.add_win('moves', game_.move_count)
                    elif game_reward == 0:
                        blue_wins.add_draw(game_.move_count)
                        red_wins.add_draw(game_.move_count)
                    else:
                        raise ValueError(f'Game reward {game_reward} unknown.')
                    break

            if simu % 10 == 0:
                print_round_results(simu, num_sims, ag_type_0, ag_type_1,
                                    blue_wins.wins, red_wins.wins)
        if save_results:
            write_results(num_sims, ag_type_0, ag_type_1,
                          red_wins, blue_wins,
                          game_times_0, game_times_1,
                          rounds_counter_per_game, rounds_counter_win_agent_0, rounds_counter_win_agent_1)
        print('')
        return red_wins.wins, blue_wins.wins, red_wins.draws


def print_round_results(i, n, ag_0, ag_1, blue_won, red_won):
    red = Fore.RED
    blue = Fore.BLUE
    rs = Style.RESET_ALL
    ag_0_res = f'Agent 0 ({blue}{ag_0}{rs})'.center(30)
    ag_1_res = f'Agent 1 ({red}{ag_1}{rs})'.center(30)
    draws = i - red_won - blue_won
    red_won = str(red_won).ljust(4)
    blue_won = str(blue_won).rjust(4)
    print(f'\r{f"Game {i}/{n}".center(10)} {ag_0_res} --> {blue_won} : {red_won} <-- {ag_1_res}'
          f'\t Draws: {draws}', end='')


def write_results(num_sims, ag_type_0, ag_type_1,
                  red_win_container, blue_win_container,
                  game_times_0, game_times_1):

    red_won = red_win_container.wins
    blue_won = blue_win_container.wins

    red_win_flags = red_win_container.win_reasons['flag']
    red_win_moves = red_win_container.win_reasons['moves']
    blue_win_flags = blue_win_container.win_reasons['flag']
    blue_win_moves = blue_win_container.win_reasons['moves']

    _, red_sum_count = red_win_container.get_count_stats()
    _, blue_sum_count = blue_win_container.get_count_stats()

    file = open("{}_vs_{}_with_{}_sims.txt".format(ag_type_0, ag_type_1, num_sims), "w")
    file.write("Statistics of {} vs. {} with {} games played.\n".format(ag_type_0, ag_type_1, num_sims))
    file.write("Overall computational time of simulation: {} seconds.\n".format(sum(game_times_0) + sum(game_times_1)))

    file.write(
        "\nAgent {} won {}/{} games (~{}%).\n".format(ag_type_0, red_won, num_sims, round(100 * red_won / num_sims, 2)))
    file.write("Reasons for winning: {} flag captures, {} wins through killing all enemies\n".format(red_win_flags,
                                                                                                     red_win_moves))

    file.write("\nAgent {} won {}/{} games (~{}%).\n".format(ag_type_1, blue_won, num_sims,
                                                             round(100 * blue_won / num_sims, 2)))
    file.write("Reasons for winning: {} flag captures, {} wins through killing all enemies\n".format(blue_win_flags,
                                                                                                     blue_win_moves))

    file.write("\nAverage game duration overall: {} rounds\n".format(round((red_sum_count + blue_sum_count) / num_sims, 2)))
    file.write("Maximum number of rounds played: {} rounds\n".format(
        max(blue_win_container.rounds_count + red_win_container.rounds_count)))
    file.write("Minimum number of rounds played: {} rounds\n".format(
        min(blue_win_container.rounds_count + red_win_container.rounds_count)))

    file.write("\nAverage game duration for {} wins: {} rounds\n".format(ag_type_0, round(
        sum(rounds_counter_win_agent_0) / len(rounds_counter_win_agent_0)), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_win_agent_0)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_win_agent_0)))

    file.write("\nAverage game duration for {} wins: {} rounds\n".format(ag_type_1, round(
        sum(rounds_counter_win_agent_1) / len(rounds_counter_win_agent_1)), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_win_agent_1)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_win_agent_1)))

    file.write("\nAverage computational time for {} wins: {} seconds\n".format(ag_type_1,
                                                                               sum(game_times_1) / len(game_times_1)))
    file.write("Maximum computational time: {} seconds\n".format(max(game_times_1)))
    file.write("Minimum computational time: {} seconds\n".format(min(game_times_1)))

    file.write("\nAverage computational time for {} wins: {} seconds\n".format(ag_type_0,
                                                                               sum(game_times_0) / len(game_times_0)))
    file.write("Maximum computational time: {} seconds\n".format(max(game_times_0)))
    file.write("Minimum computational time: {} seconds\n".format(min(game_times_0)))
    file.close()


if __name__ == '__main__':
    arena = Arena(agent.MiniMax(0), agent.MiniMax(1), game_size="small")
    arena.pit(num_sims=100, show_game=False)

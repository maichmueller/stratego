############################################################################
#
# The whole project is written by Leonhard Bereska and Michael Aichmüller.
# It is hard, if not almost impossible to try to differentiate between the
# parts coded by Bereska and the parts by Aichmüller, keep the vague
# scheme in mind:
# The whole game code was a 50/50 split (legal_moves, do_move, helpers etc.)
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

import numpy as np
import game
import pieces
import agent
import env
import helpers
from timeit import default_timer as timer
import re


def draw_random_setup(types_available, team, game_dim):
    """
    Draw a random setup from the set of types types_available after placing the flag
    somewhere in the last row of the board of the side of 'team', or behind the obstacle.
    :param types_available: list of types to draw from, integers
    :param team: boolean, 1 or 0 depending on the team
    :return: the setup, in numpy array form
    """

    nr_pieces = len(types_available)-1
    types_available = [type_ for type_ in types_available if not type_ == 0]
    if game_dim == 5:
        row_offset = 2
    elif game_dim == 7:
        row_offset = 3
    else:
        row_offset = 4
    setup_agent = np.empty((row_offset, game_dim), dtype=object)
    if team == 0:
        flag_positions = [(game_dim-1, j) for j in range(game_dim)]
        flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
        flag_pos = game_dim-1 - flag_positions[flag_choice][0], game_dim-1 - flag_positions[flag_choice][1]
        setup_agent[flag_pos] = pieces.Piece(0, 0, flag_positions[flag_choice])

        types_draw = np.random.choice(types_available, nr_pieces, replace=False)
        positions_agent_0 = [(i, j) for i in range(game_dim-row_offset, game_dim) for j in range(game_dim)]
        positions_agent_0.remove(flag_positions[flag_choice])

        for idx in range(nr_pieces):
            pos = positions_agent_0[idx]
            setup_agent[(game_dim-1 - pos[0], game_dim-1 - pos[1])] = pieces.Piece(types_draw[idx], 0, pos)
    elif team == 1:
        flag_positions = [(0, j) for j in range(game_dim)]
        flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
        setup_agent[flag_positions[flag_choice]] = pieces.Piece(0, 1, flag_positions[flag_choice])

        types_draw = np.random.choice(types_available, nr_pieces, replace=False)
        positions_agent_1 = [(i, j) for i in range(row_offset) for j in range(game_dim)]
        positions_agent_1.remove(flag_positions[flag_choice])

        for idx in range(nr_pieces):
            pos = positions_agent_1[idx]
            setup_agent[pos] = pieces.Piece(types_draw[idx], 1, pos)
    return setup_agent


def simulation(game_, num_simulations, setup_0=None, setup_1=None, show_game=False):
    """
    Simulate num_simulations many games of the agent of type agent_type_0 against the agent of
    type agent_type_1. If setup_0 or setup_1 are provided respectively, then take the pieces
    setup from those. If show_game is True, the game will be printed by the internal function.
    :param game_: game object that runs the simulation
    :param num_simulations: integer number of games to simulate
    :param setup_0: (optional) numpy array of the setup of agent 0
    :param setup_1: (optional) numpy array of the setup of agent 1
    :param show_game: (optional) boolean, whether to show the game or not
    :return: None, writes results to a file named after the agents acting
    """
    blue_won = 0
    blue_wins_bc_flag = 0
    blue_wins_bc_noMovesLeft = 0
    red_won = 0
    red_wins_bc_flag = 0
    red_wins_bc_noMovesLeft = 0
    rounds_counter_per_game = []
    rounds_counter_win_agent_0 = []
    rounds_counter_win_agent_1 = []

    game_times_0 = []
    game_times_1 = []
    types = game_.types_available
    for simu in range(num_simulations):  # simulate games
        # reset setup with new setup if none given
        if setup_0 is not None:
            setup_agent_0 = setup_0
        else:
            setup_agent_0 = draw_random_setup(types, 0, game_.game_dim)
        if setup_1 is not None:
            setup_agent_1 = setup_1
        else:
            setup_agent_1 = draw_random_setup(types, 1, game_.game_dim)
        game_.agents[0].setup = setup_agent_0
        game_.agents[1].setup = setup_agent_1
        game_.reset()

        agent_output_type_0 = str(game_.agents[0])
        agent_output_type_1 = str(game_.agents[1])
        agent_output_type_0 = re.search('agent.(.+?) object', agent_output_type_0).group(1)
        agent_output_type_1 = re.search('agent.(.+?) object', agent_output_type_1).group(1)

        game_time_s = timer()
        if (simu+1) % 1 == 0:
            print('{} won: {}, {} won: {}, Game {}/{}'.format(agent_output_type_0,
                                                              red_won,
                                                              agent_output_type_1,
                                                              blue_won, simu,
                                                              num_simulations))
            print('{} won by flag capture: {}, {} won by moves: {}, Game {}/{}'.format(agent_output_type_0,
                                                                                       red_wins_bc_flag,
                                                                                       agent_output_type_0,
                                                                                       red_wins_bc_noMovesLeft,
                                                                                       simu,
                                                                                       num_simulations))
            print('{} won by flag capture: {}, {} won by moves: {}, Game {}/{}'.format(agent_output_type_1,
                                                                                       blue_wins_bc_flag,
                                                                                       agent_output_type_1,
                                                                                       blue_wins_bc_noMovesLeft,
                                                                                       simu,
                                                                                       num_simulations))
        print("Game number: " + str(simu + 1))
        for step in range(2000):
            if show_game:
                helpers.print_board(game_.board)
            game_reward = game_.run_step()
            if game_reward is not None:
                if game_reward[0] == 1:  # count wins
                    game_times_0.append(timer() - game_time_s)
                    red_won += 1
                    red_wins_bc_flag += 1
                    rounds_counter_win_agent_0.append(game_.move_count)
                elif game_reward[0] == 2:
                    game_times_0.append(timer() - game_time_s)
                    red_won += 1
                    red_wins_bc_noMovesLeft += 1
                    rounds_counter_win_agent_0.append(game_.move_count)
                elif game_reward[0] == -1:
                    game_times_1.append(timer() - game_time_s)
                    blue_won += 1
                    blue_wins_bc_flag += 1
                    rounds_counter_win_agent_1.append(game_.move_count)
                else:
                    game_times_1.append(timer() - game_time_s)
                    blue_won += 1
                    blue_wins_bc_noMovesLeft += 1
                    rounds_counter_win_agent_1.append(game_.move_count)
                rounds_counter_per_game.append(game_.move_count)
                break
        if show_game:
            helpers.print_board(game_.board)
    file = open("{}_vs_{}_with_{}_sims.txt".format(agent_output_type_0, agent_output_type_1, num_simulations), "w")
    file.write("Statistics of {} vs. {} with {} games played.\n".format(agent_output_type_0, agent_output_type_1, num_simulations))
    file.write("Overall computational time of simulation: {} seconds.\n".format(sum(game_times_0) + sum(game_times_1)))

    file.write("\nAgent {} won {}/{} games (~{}%).\n".format(agent_output_type_0, red_won, num_simulations, round(100*red_won/num_simulations, 2)))
    file.write("Reasons for winning: {} flag captures, {} wins through killing all enemies\n".format(red_wins_bc_flag, red_wins_bc_noMovesLeft))

    file.write("\nAgent {} won {}/{} games (~{}%).\n".format(agent_output_type_1, blue_won, num_simulations, round(100*blue_won/num_simulations, 2)))
    file.write("Reasons for winning: {} flag captures, {} wins through killing all enemies\n".format(blue_wins_bc_flag, blue_wins_bc_noMovesLeft))

    file.write("\nAverage game duration overall: {} rounds\n".format(round(sum(rounds_counter_per_game)/num_simulations), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_per_game)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_per_game)))

    file.write("\nAverage game duration for {} wins: {} rounds\n".format(agent_output_type_0, round(sum(rounds_counter_win_agent_0)/len(rounds_counter_win_agent_0)), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_win_agent_0)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_win_agent_0)))

    file.write("\nAverage game duration for {} wins: {} rounds\n".format(agent_output_type_1, round(sum(rounds_counter_win_agent_1)/len(rounds_counter_win_agent_1)), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_win_agent_1)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_win_agent_1)))

    file.write("\nAverage computational time for {} wins: {} seconds\n".format(agent_output_type_1, sum(game_times_1)/len(game_times_1)))
    file.write("Maximum computational time: {} seconds\n".format(max(game_times_1)))
    file.write("Minimum computational time: {} seconds\n".format(min(game_times_1)))

    file.write("\nAverage computational time for {} wins: {} seconds\n".format(agent_output_type_0, sum(game_times_0)/len(game_times_0)))
    file.write("Maximum computational time: {} seconds\n".format(max(game_times_0)))
    file.write("Minimum computational time: {} seconds\n".format(min(game_times_0)))
    file.close()
    return

#simulation(agent_type_0="minmax", agent_type_1="heuristic", num_simulations=1000)

def simu_env(env, num_simulations=1000, watch=True):
    """
    Plots simulated games in an environment for visualization
    :param env: environment to be run
    :param n_runs: how many episodes should be run
    :param watch: do you want to plot the game (watch=True) or just see the results (watch=False)?
    :return: plot of each step in the environment
    """
    blue_won = 0
    blue_wins_bc_flag = 0
    blue_wins_bc_noMovesLeft = 0
    red_won = 0
    red_wins_bc_flag = 0
    red_wins_bc_noMovesLeft = 0
    rounds_counter_per_game = []
    rounds_counter_win_agent_0 = []
    rounds_counter_win_agent_1 = []
    ties = 0
    env_type = str(env)
    agent_output_type_0 = str(env.agents[0])
    agent_output_type_1 = str(env.agents[1])
    env_type = re.search('env.(.+?) object', env_type).group(1)
    agent_output_type_0 = re.search('agent.(.+?) object', agent_output_type_0).group(1)
    agent_output_type_1 = re.search('agent.(.+?) object', agent_output_type_1).group(1)
    game_times_0 = []
    game_times_1 = []
    for simu in range(num_simulations):
        print('{} won: {}, {} won: {}, ties: {}, Game {}/{}'.format(agent_output_type_0,
                                                          red_won,
                                                          agent_output_type_1,
                                                          blue_won,
                                                          ties, simu, num_simulations))
        print('{} won by flag capture: {}, {} won by moves: {}, Game {}/{}'.format(agent_output_type_0,
                                                                                   red_wins_bc_flag,
                                                                                   agent_output_type_0,
                                                                                   red_wins_bc_noMovesLeft,
                                                                                   simu,
                                                                                   num_simulations))
        print('{} won by flag capture: {}, {} won by moves: {}, Game {}/{}'.format(agent_output_type_1,
                                                                                   blue_wins_bc_flag,
                                                                                   agent_output_type_1,
                                                                                   blue_wins_bc_noMovesLeft,
                                                                                   simu,
                                                                                   num_simulations))
        print("Game number: {}".format(simu+1))
        game_time_s = timer()
        env.reset()
        # environment.show()
        done = False
        while not done:
            _, done, won = env.step()
            if watch:
                env.show()
            if done:
                if won == 1:
                    game_times_0.append(timer() - game_time_s)
                    red_won += 1
                    red_wins_bc_flag += 1
                    rounds_counter_win_agent_0.append(env.move_count)
                elif won == 2:
                    game_times_0.append(timer() - game_time_s)
                    red_won += 1
                    red_wins_bc_noMovesLeft += 1
                    rounds_counter_win_agent_0.append(env.move_count)
                elif won == -1:
                    game_times_1.append(timer() - game_time_s)
                    blue_won += 1
                    blue_wins_bc_flag += 1
                    rounds_counter_win_agent_1.append(env.move_count)
                elif won == -2:
                    game_times_1.append(timer() - game_time_s)
                    blue_won += 1
                    blue_wins_bc_noMovesLeft += 1
                    rounds_counter_win_agent_1.append(env.move_count)
                rounds_counter_per_game.append(env.move_count)

            elif env.steps > 100:  # break game that takes too long
                ties += 1
                break
    file = open("{}_vs_{}_with_{}_sims_{}.txt".format(agent_output_type_0, agent_output_type_1, num_simulations, env_type), "w")
    file.write("Statistics of {} vs. {} with {} games played.\n".format(agent_output_type_0, agent_output_type_1, num_simulations))
    file.write("Overall computational time of simulation: {} seconds.\n".format(sum(game_times_0) + sum(game_times_1)))

    file.write("\nAgent {} won {}/{} games (~{}%).\n".format(agent_output_type_0, red_won, num_simulations, round(100*red_won/num_simulations, 2)))
    file.write("Reasons for winning: {} flag captures, {} wins through killing all enemies\n".format(red_wins_bc_flag, red_wins_bc_noMovesLeft))

    file.write("\nAgent {} won {}/{} games (~{}%).\n".format(agent_output_type_1, blue_won, num_simulations, round(100*blue_won/num_simulations, 2)))
    file.write("Reasons for winning: {} flag captures, {} wins through killing all enemies\n".format(blue_wins_bc_flag, blue_wins_bc_noMovesLeft))

    file.write("\nNumber of tied games: {}\n".format(ties))

    file.write("\nAverage game duration overall: {} rounds\n".format(round(sum(rounds_counter_per_game)/num_simulations), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_per_game)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_per_game)))

    file.write("\nAverage game duration for {} wins: {} rounds\n".format(agent_output_type_0, round(sum(rounds_counter_win_agent_0)/len(rounds_counter_win_agent_0)), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_win_agent_0)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_win_agent_0)))

    file.write("\nAverage game duration for {} wins: {} rounds\n".format(agent_output_type_1, round(sum(rounds_counter_win_agent_1)/len(rounds_counter_win_agent_1)), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_win_agent_1)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_win_agent_1)))

    file.write("\nAverage computational time for {} wins: {} seconds\n".format(agent_output_type_1, sum(game_times_1)/len(game_times_1)))
    file.write("Maximum computational time: {} seconds\n".format(max(game_times_1)))
    file.write("Minimum computational time: {} seconds\n".format(min(game_times_1)))

    file.write("\nAverage computational time for {} wins: {} seconds\n".format(agent_output_type_0, sum(game_times_0)/len(game_times_0)))
    file.write("Maximum computational time: {} seconds\n".format(max(game_times_0)))
    file.write("Minimum computational time: {} seconds\n".format(min(game_times_0)))
    file.close()



# for testing the learners and the smaller environments
#environment = env.Stratego(agent.Heuristic(0), agent.MiniMax(1))
#simu_env(environment, num_simulations=1000, watch=False)

#for testing the full game (can use different setup functions)
game_ = game.Game(agent.Random(0), agent.Random(1), game_size="big")
simulation(game_, num_simulations=1000, show_game=False)

from stratego.agent import Agent
import numpy as np
from scipy import spatial
import torch

import copy
from stratego import utils
from stratego.engine import Logic


class MiniMax(Agent):
    """
    Agent deciding his moves based on the minimax algorithm. The agent guessed the enemies setup
    before making a decision by using the current information available about the pieces.
    """

    def __init__(self, team, depth=None):
        super(MiniMax, self).__init__(team=team)
        # rewards for planning the move
        self.kill_reward = 10  # killing an enemy piece
        self.neutral_fight = 2  # a neutral outcome of a fight
        self.winGameReward = 100  # finding the enemy flag
        self.certainty_multiplier = 1.2  # killing known, not guessed, enemy pieces

        # initial maximum depth of the minimax algorithm
        self.ext_depth = depth
        self.max_depth = 2  # standard max depth

        # the matrix table for deciding battle outcomes between two pieces
        self.battleMatrix = utils.get_bm()

    def decide_move(self, state, logic: Logic = Logic()):
        """
        Depending on the amount of enemy pieces left, we are entering the start, mid or endgame
        and planning through the minimax algorithm.
        :return: tuple of tuple positions representing the move
        """
        if self.ext_depth is None:
            self.set_max_depth()  # set max_depth each turn
        else:
            self.max_depth = self.ext_depth
        # make sure a flag win will be discounted by a factor that guarantees a preference towards immediate flag kill
        self.winGameReward = max(self.winGameReward, self.max_depth * self.kill_reward)
        return self.minimax(max_depth=self.max_depth)

    def set_max_depth(self):
        n_alive_enemies = sum(
            [True for piece in self.ordered_opp_pieces if not piece.dead]
        )
        if 7 < n_alive_enemies <= 10:
            # one move each player lookahead
            self.max_depth = 2
        elif 4 <= n_alive_enemies <= 7:
            # two moves each player lookahead
            self.max_depth = 4
        elif n_alive_enemies <= 3:
            # four moves each player lookahead
            self.max_depth = 8

    def minimax(self, max_depth):
        """
        given the maximum depth, copy the known board so far, assign the pieces by random, while still
        respecting the current knowledge, and then decide the move via minimax algorithm.
        :param max_depth: int
        :return: tuple of spatial tuples
        """
        curr_board = copy.deepcopy(self.board)
        curr_board = self.draw_consistent_enemy_setup(curr_board)
        chosen_action = self.max_val(
            curr_board, 0, -float("inf"), float("inf"), max_depth
        )[1]
        return chosen_action

    def max_val(self, board, current_reward, alpha, beta, depth):
        """
        Do the max players step in the minimax algorithm. Check first if the given board is in
        a terminal state. If not, we will do each possible move once and send the process to
        min_val to do the min players step.
        :param board: the current board, numpy array
        :param current_reward: the current value the path has accumulated
        :param alpha: alpha threshold of the minimax alg
        :param beta: beta threshold of the minimax alg
        :param depth: the depth the process is at, integer
        :return: tuple of best value, and a associated best_action (float, tuple)
        """
        # this is what the expectimax agent will think

        # get my possible actions, then shuffle them to ensure randomness when no action
        # stands out as the best
        my_doable_actions = utils.get_poss_moves(board, self.team)
        np.random.shuffle(my_doable_actions)

        # check for terminal-state scenario
        done, won = self.goal_test(my_doable_actions, board, max_val=True)
        if done or depth == 0:
            return current_reward + self.get_terminal_reward(done, won, depth), None

        val = -float("inf")
        best_action = None
        for action in my_doable_actions:
            board, fight_result = self.do_move(
                action, board=board, bookkeeping=False, true_gameplay=False
            )
            temp_reward = current_reward + self.add_temp_reward(fight_result)
            new_val = self.min_val(board, temp_reward, alpha, beta, depth - 1)[0]
            if val < new_val:
                val = new_val
                best_action = action
            if val >= beta:
                self.undo_last_move(board)
                best_action = action
                return val, best_action
            alpha = max(alpha, val)
            board = self.undo_last_move(board)
        return val, best_action

    def min_val(self, board, current_reward, alpha, beta, depth):
        """
        Step of the minimizing player in the minimax algorithm. See max_val for documentation.
        """
        # this is what the opponent will think, the min-player

        # get my possible actions, then shuffle them to ensure randomness when no action
        # stands out as the best
        my_doable_actions = utils.get_poss_moves(board, self.other_team)
        np.random.shuffle(my_doable_actions)

        # check for terminal-state scenario or maximum depth
        done, won = self.goal_test(my_doable_actions, board, max_val=False)
        if done or depth == 0:
            return current_reward + self.get_terminal_reward(done, won, depth), None

        val = float("inf")  # initial value set, so min comparison later possible
        best_action = None
        # iterate through all actions
        for action in my_doable_actions:
            board, fight_result = self.do_move(
                action, board=board, bookkeeping=False, true_gameplay=False
            )
            temp_reward = current_reward - self.add_temp_reward(fight_result)
            new_val = self.max_val(board, temp_reward, alpha, beta, depth - 1)[0]
            if val > new_val:
                val = new_val
                best_action = action
            if val <= alpha:
                self.undo_last_move(board)
                return val, best_action
            beta = min(beta, val)
            board = self.undo_last_move(board)
        return val, best_action

    def add_temp_reward(self, fight_result):
        """
        reward the fight given the outcome of it.
        :param fight_result: integer category of the fight outcome
        :return: reward, float
        """
        # depending on the fight we want to update the current paths value
        temp_reward = 0
        if fight_result is not None:
            if fight_result == 1:  # attacker won
                temp_reward = self.kill_reward
            elif fight_result == 2:  # attacker won, every piece was known before
                temp_reward = int(self.certainty_multiplier * self.kill_reward)
            elif fight_result == 0:  # neutral outcome
                temp_reward = self.neutral_fight  # both pieces die
            elif fight_result == -1:  # attacker lost
                temp_reward = -self.kill_reward
            elif fight_result == -2:  # attacker lost, every piece was known before
                temp_reward = -int(self.certainty_multiplier * self.kill_reward)
        return temp_reward

    def goal_test(self, actions_possible, board, max_val):
        """
        check the board for whether a flag has been captured already and return the winning game rewards,
        if not check whether there are no actions possible anymore, return TRUE then, or FALSE.
        :param actions_possible: list of moves
        :param board: numpy array (5, 5)
        :param max_val: boolean, decider whether this is a goal test for maximizing player
        :return: boolean: reached terminal state, boolean: own team (True) or other team won (False)
        """
        flag_alive = [False, False]
        for pos, piece in np.ndenumerate(board):
            if piece is not None and int == 0:
                flag_alive[piece.team] = True
        if not flag_alive[self.other_team]:
            return True, True
        # if not flag_alive[self.team]:
        #     return True, False
        if not actions_possible:
            # print('cannot move anymore')
            if (
                max_val
            ):  # the minmax agent is the one doing max_val, so if he cant move -> loss for him
                won = False
            else:
                won = True
            return True, won
        else:
            return False, None

    def get_terminal_reward(self, done, won, depth):
        """
        Reward for ending the game on a certain depth. If ended because of flag capture, then
        reward with a depth discounted winGameReward. If ended because of depth limitation,
        return 0
        :param done: boolean, indicate whether the game ended
        :param won: boolean, indicate whether the game was won or lost
        :param depth: the depth at which the game ended
        :return: game end reward, float
        """
        if not done:
            return 0
        else:
            if won:
                terminal_reward = self.winGameReward
            else:
                terminal_reward = -self.winGameReward
            return (
                terminal_reward
                * (depth + 1)
                / (self.max_depth + 1)
                * (terminal_reward / self.kill_reward)
            )

    def update_prob_by_fight(self, enemy_piece):
        """
        update the information about the given piece, after a fight occured
        :param enemy_piece: object of class Piece
        :return: change is in-place, no value specified
        """
        enemy_piece.potential_types = [enemy_int]

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
                for idx, type in enumerate(moving_piece.potential_types)
                if type in [0, 11]
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

    def undo_last_move(self, board):
        """
        Undo the last move in the memory. Return the updated board.
        :param board: numpy array (5, 5)
        :return: board
        """
        last_move = self.last_N_moves.pop_last()
        if last_move is None:
            raise ValueError("No last move to undo detected!")
        before_piece = self.pieces_last_N_Moves_beforePos.pop_last()
        board[last_move[0]] = before_piece
        # the piece at the 'before' spatial was the one that moved, so needs its
        # last entry in the move history deleted
        before_piece.position = last_move[0]
        board[last_move[1]] = self.pieces_last_N_Moves_afterPos.pop_last()
        return board


class OmniscientMiniMax(MiniMax):
    """
    Child of MiniMax agent. This agent is omniscient and thus knows the location and type of each
    piece of the enemy. It then plans by doing a minimax algorithm.
    """

    def __init__(self, team, setup=None, depth=None):
        super(OmniscientMiniMax, self).__init__(team=team, setup=setup, depth=depth)

    def minimax(self, max_depth):
        chosen_action = self.max_val(
            self.board, 0, -float("inf"), float("inf"), max_depth
        )[1]
        return chosen_action


class OmniscientHeuristic(OmniscientMiniMax):
    """
    Omniscient Minimax planner, that uses a learned board heuristic as evaluation function.
    """

    def __init__(self, team, setup=None):
        super(OmniscientHeuristic, self).__init__(team=team, setup=setup)
        self.evaluator = Stratego(team)

    def get_network_reward(self):
        state = self.evaluator.state_to_tensor()
        self.evaluator.network.eval()
        state_action_values = self.evaluator.network(torch.tensor(state)).data.numpy()
        return np.max(state_action_values)

    def get_terminal_reward(self, done, won, depth):
        if not done:
            return self.get_network_reward()
        else:
            if won:
                terminal_reward = self.winGameReward
            else:
                terminal_reward = -self.winGameReward
            return (
                terminal_reward
                * (depth + 1)
                / (self.max_depth + 1)
                * (terminal_reward / self.kill_reward)
            )


class Heuristic(MiniMax):
    """
    Non omniscient Minimax planner with learned board evluation function.
    """

    def __init__(self, team, setup=None):
        super(Heuristic, self).__init__(team=team, setup=setup)
        self.evaluator = Stratego(team)

    def get_network_reward(self):
        state = self.evaluator.state_to_tensor()
        self.evaluator.network.eval()
        state_action_values = self.evaluator.network(torch.tensor(state)).data.numpy()
        return np.max(state_action_values)

    def get_terminal_reward(self, done, won, depth):
        if not done:
            return self.get_network_reward()
        else:
            if won:
                terminal_reward = self.winGameReward
            elif not won:
                terminal_reward = -self.winGameReward
            return (
                terminal_reward
                * (depth + 1)
                / (self.max_depth + 1)
                * (terminal_reward / self.kill_reward)
            )


class MonteCarlo(MiniMax):
    """
    Monte carlo agent, simulating the value of each move and choosing the best.
    """

    def __init__(self, team, setup=None, number_of_iterations_game_sim=40):
        super(MonteCarlo, self).__init__(team=team, setup=setup)
        self._nr_iterations_of_game_sim = number_of_iterations_game_sim
        self._nr_of_max_turn_sim = 15
        self._nr_of_enemy_setups_to_draw = 20

    def decide_move(self):
        """
        given the maximum depth, copy the known board so far, assign the pieces by random, while still
        respecting the current knowledge, and then decide the move via minimax algorithm.
        :return: tuple of spatial tuples
        """
        possible_moves = utils.get_poss_moves(self.board, self.team)
        next_action = None
        if possible_moves:
            values_of_moves = dict.fromkeys(possible_moves, 0)
            for move in possible_moves:
                for draw in range(self._nr_of_enemy_setups_to_draw):
                    curr_board = self.draw_consistent_enemy_setup(
                        copy.deepcopy(self.board)
                    )
                    curr_board, _ = self.do_move(
                        move, curr_board, bookkeeping=False, true_gameplay=False
                    )
                    values_of_moves[move] += (
                        self.approximate_value_of_board(curr_board)
                        / self._nr_of_enemy_setups_to_draw
                    )
                    self.undo_last_move(curr_board)
            evaluations = list(values_of_moves.values())
            actions = list(values_of_moves.keys())
            next_action = actions[evaluations.index(max(evaluations))]
        return next_action

    def approximate_value_of_board(self, board):
        """
        Simulate the game to the max number of turns a lot of times and evaluating the simulation
        by whether he won and how many more pieces he has left than the opponent.
        :param board:
        :return:
        """
        finished = False
        turn = 0
        evals = []
        for i in range(self._nr_iterations_of_game_sim):
            board_copy = copy.deepcopy(board)
            while not finished:
                actions = utils.get_poss_moves(board_copy, turn)
                if actions:  # as long as actions are left to be done, we do them
                    move = np.random.choice(actions)
                    board_copy, _ = self.do_move(move, board_copy)
                # check whether the game is terminal
                done, won = self.goal_test(actions, board_copy, turn)
                if done:
                    # if terminal, calculate the bonus we want to reward this simulation with
                    my_team = self.get_team_from_board(board, self.team)
                    enemy_team = self.get_team_from_board(board, self.other_team)
                    bonus = (len(my_team) - len(enemy_team)) / 20
                    # -1+2*won equals -1+2*0=-1 for won=False, and -1+2*1=1 for won=True
                    # bonus is negative if enemy team has more pieces
                    evals.append(-1 + 2 * won + bonus)
                    finished = True
                elif (
                    turn > self._nr_of_max_turn_sim
                ):  # check if we reached the max number of turns
                    # calculate bonus
                    my_team = self.get_team_from_board(board, self.team)
                    enemy_team = self.get_team_from_board(board, self.other_team)
                    bonus = (len(my_team) - len(enemy_team)) / 20
                    # -1+2*won equals -1+2*0=-1 for won=False, and -1+2*1=1 for won=True
                    # bonus is negative if enemy team has more pieces
                    evals.append(bonus)
                    finished = True
                turn = (turn + 1) % 2
        return sum(evals) / len(evals)

    def get_team_from_board(self, board, team):
        team_list = []
        for pos, piece in np.ndenumerate(board):
            if piece is not None and piece.team == team:
                team_list.append(piece)
        return team_list

    def goal_test(self, actions_possible, board, turn):
        """
        check the board for whether a flag has been captured already and return the winning game rewards,
        if not check whether there are no actions possible anymore, return TRUE then, or FALSE.
        :param actions_possible: list of moves
        :param board: numpy array (5, 5)
        :return: boolean: reached terminal state, boolean: own team (True) or other team won (False)
        """
        # if board is not None:
        flag_alive = [False, False]
        for pos, piece in np.ndenumerate(board):
            if piece is not None and int == 0:
                flag_alive[piece.team] = True
        if not flag_alive[self.other_team]:
            return True, True
        if not flag_alive[self.team]:
            return True, False
        if not actions_possible:
            if turn == self.team:
                return True, False
            else:
                return True, True
        else:
            return False, None


class MonteCarloHeuristic(MonteCarlo):
    """
    Monte Carlo agent that evaluates boards not by simulating the game, but by taking the heuristic
    from a learner.
    """

    def __init__(self, team, setup=None):
        super(MonteCarloHeuristic, self).__init__(
            team=team, setup=setup, number_of_iterations_game_sim=1
        )
        self.evaluator = Stratego(team)

    def get_network_reward(self):
        state = self.evaluator.state_to_tensor()
        self.evaluator.network.eval()
        state_action_values = self.evaluator.network(torch.tensor(state)).data.numpy()
        return np.max(state_action_values)

    def decide_move(self):
        """
        given the maximum depth, copy the known board so far, assign the pieces by random, while still
        respecting the current knowledge, and then decide the move via minimax algorithm.
        :return: tuple of spatial tuples
        """
        possible_moves = utils.get_poss_moves(self.board, self.team)
        next_action = None
        if possible_moves:
            values_of_moves = dict.fromkeys(possible_moves, 0)
            for move in possible_moves:
                for draw in range(self._nr_of_enemy_setups_to_draw):
                    curr_board = self.draw_consistent_enemy_setup(
                        copy.deepcopy(self.board)
                    )
                    curr_board, _ = self.do_move(
                        move, curr_board, bookkeeping=False, true_gameplay=False
                    )
                    self.evaluator.board = curr_board
                    values_of_moves[move] += (
                        self.get_network_reward() / self._nr_of_enemy_setups_to_draw
                    )
                    self.undo_last_move(curr_board)
            evaluations = list(values_of_moves.values())
            actions = list(values_of_moves.keys())
            next_action = actions[evaluations.index(max(evaluations))]
        return next_action

import random
import numpy as np
import pieces
import copy
# from collections import Counter
from scipy import spatial
# from scipy import optimize
from cythonized import utils
from cythonized.utils import GLOBAL_DEVICE
import torch
import models

import argparse
import os
import shutil
import time
import random
import math
import sys
from game import GameState
from collections import defaultdict

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import Counter
import abc


class Agent:
    """
    Agent decides which action to take
    """
    def __init__(self, team, setup=None):
        self.team = team
        self.other_team = (self.team + 1) % 2
        self.setup = setup
        self.board = None

        self.learner = False

        self.move_count = 0  # round counter of the game
        self.last_N_moves = []
        self.pieces_last_N_Moves_beforePos = []
        self.pieces_last_N_Moves_afterPos = []

        self.battleMatrix = utils.get_bm()

        # list of enemy pieces
        self.ordered_opp_pieces = []  # TODO replace this?

    def install_board(self, board, reset=False):
        """
        Install the opponents setup in this agents board and evaluate information for potential
        minmax operations later on
        :param opp_setup: numpy array of any shape with pieces objects stored in them
        :return: None
        """
        self.board = copy.deepcopy(board)
        enemy_pieces = []
        self.own_pieces = []
        for idx, piece in np.ndenumerate(self.board):
            if piece is not None:
                if piece.team == self.team:
                    self.own_pieces.append(piece)
                elif piece.team == self.other_team:
                    enemy_pieces.append(piece)
        enemy_types = [piece.type for piece in enemy_pieces]

        self.ordered_opp_pieces = enemy_pieces
        self.own_pieces = self.own_pieces

        for idx, piece in enumerate(self.ordered_opp_pieces):
            piece.potential_types = copy.copy(enemy_types)
            piece.hidden = True
        if reset:
            self.move_count = 0  # round counter of the game
            self.last_N_moves = []
            self.pieces_last_N_Moves_beforePos = []
            self.pieces_last_N_Moves_afterPos = []
        self.action_represent()  # for installing action representation in reinforcement agents
        return None

    def action_represent(self):  # does nothing but is important for Reinforce
        return

    def update_board(self, updated_piece, board=None):
        """
        update the current board at the position with the piece both given by updated_piece
        :param updated_piece: tuple of (position, piece) with position being a tuple and piece being
        of class piece
        :param board: numpy array of shape (int, int) representing the board that should be updated.
        Default is the agent's own board
        :return: Updates are made in-place. No return value specified
        """
        if board is None:
            board = self.board
        if updated_piece[1] is not None:
            updated_piece[1].change_position(updated_piece[0])
        board[updated_piece[0]] = updated_piece[1]

    def decide_move(self, *args, **kwargs):
        """
        Implementation of the agent's move for the current round
        :return: tuple of "from" position tuple to "to" position tuple representing the move
        """
        raise NotImplementedError

    def do_move(self, move, board=None, bookkeeping=True, true_gameplay=False):
        """
        Execute the given move on the given board. If bookkeeping is TRUE, any dying pieces will be
        collected by the agent and stored for evaluation. If true_gameplay is TRUE, any moves made are
        due to the real game having proceeded, if FALSE it is considered a move made during planning (e.g
        minmax calculations). Return the new board and the outcome of any occuring fights during the move.
        :param move: tuple of positions tuple "from", "to" in the range of (0..4, 0..4)
        :param board: numpy array (5, 5) holding pieces objects
        :param bookkeeping: boolean
        :param true_gameplay: boolean
        :return: tuple of (numpy array (5, 5), integer)
        """
        from_ = move[0]
        to_ = move[1]
        turn = self.move_count % 2
        fight_outcome = None
        if board is None:
            board = self.board
            board[from_].has_moved = True
        moving_piece = board[from_]
        attacked_field = board[to_]
        self.last_N_moves.append(move)
        self.pieces_last_N_Moves_afterPos.append(attacked_field)
        self.pieces_last_N_Moves_beforePos.append(moving_piece)
        if not board[to_] is None:  # Target field is not empty, then has to fight
            if board is None:
                # only uncover them when the real board is being played on
                attacked_field.hidden = False
                moving_piece.hidden = False
            fight_outcome = self.fight(moving_piece, attacked_field, collect_dead_pieces=bookkeeping)
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board((to_, moving_piece), board=board)
                self.update_board((from_, None), board=board)
            elif fight_outcome == 0:
                self.update_board((to_, None), board=board)
                self.update_board((from_, None), board=board)
            else:
                self.update_board((from_, None), board=board)
            if true_gameplay:
                if turn == self.team:
                    self.update_prob_by_fight(attacked_field)
                else:
                    self.update_prob_by_fight(moving_piece)
        else:
            self.update_board((to_, moving_piece), board=board)
            self.update_board((from_, None), board=board)
            if true_gameplay:
                if turn == self.other_team:
                    self.update_prob_by_move(move, moving_piece)
        return board, fight_outcome

    def fight(self, piece_att, piece_def, collect_dead_pieces=True):
        """
        Determine the outcome of a fight between the provided attacking piece and the defening piece.
        If collect_dead_pieces is TRUE, we will store the fallen pieces in the dict.
        Return the outcome of the fight, categorized by an integer (1=ATTACKER won, 0=TIE, -1=DEFENDER won),
        double the outcome if any of the pieces' types were not known before, but guessed
        (done for minmax calculation).
        :param piece_att: object of class Piece
        :param piece_def: object of class Piece
        :param collect_dead_pieces: boolean
        :return: integer
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if collect_dead_pieces:
            if outcome == 1:
                piece_def.dead = True
            elif outcome == 0:
                piece_def.dead = True
                piece_att.dead = True
            elif outcome == -1:
                piece_att.dead = True
            if piece_att.guessed or piece_def.guessed:
                outcome *= 2
            return outcome
        elif piece_att.guessed or piece_def.guessed:
            outcome *= 2
        return outcome

    def update_prob_by_fight(self, *args):
        pass

    def update_prob_by_move(self, *args):
        pass


class Random(Agent):
    """
    Agent who chooses his actions at random
    """
    def __init__(self, team, setup=None):
        super(Random, self).__init__(team=team, setup=setup)

    def install_board(self, board, reset=False):
        super().install_board(board, reset=reset)
        for piece in self.own_pieces + self.ordered_opp_pieces:
            piece.hidden = False

    def decide_move(self):
        actions = utils.get_poss_moves(self.board, self.team)
        if not actions:
            return None
        else:
            return random.choice(actions)


class Reinforce(Agent, abc.ABC):
    """
    Agent approximating action-value functions with an artificial neural network
    trained with Q-learning
    """
    global GLOBAL_DEVICE

    def __init__(self, team, setup=None):
        super(Reinforce, self).__init__(team=team, setup=setup)
        self.learner = True
        self.state_dim = None
        self.action_dim = None
        self.model = None
        self.actors = None
        self.actions = None
        self.relation_dict = None
        self.reward = 0

    def install_board(self, board, reset=False):
        super().install_board(board, reset=False)
        opp_types = sorted(set([piece.type for piece in self.ordered_opp_pieces]))
        for piece in self.ordered_opp_pieces:
            piece.potential_types = opp_types
        for piece in self.own_pieces:
            piece.hidden = False

    def decide_move(self):
        board = self.draw_consistent_enemy_setup(copy.deepcopy(self.board))
        state = self.board_to_state(board)
        action = self.select_action(state)
        if action is not None:
            move = self.action_to_move(action)
        else:
            return None
        return move

    def state_represent(self):
        """
        Specify the state representation as input for the network
        """
        return NotImplementedError

    def set_action_rep(self, actors, actions, relation_dict):
        """
        Install the action representation given by the game.
        """
        self.actors = []
        for actor, count in Counter(actors).items():
            type_, version = list(map(int, actor.split('_')))
            for piece in self.board.flatten():
                if piece is not None and piece.type == type_ and piece.version == version:
                    self.actors += [piece] * count

        self.actions = actions
        self.relation_dict = relation_dict

    def select_action(self, state):

        self.model.eval()
        state_action_values = self.model(state).view(-1)
        # state_action_values = state_action_values.cpu()

        action = self.actions[int(torch.argmax(state_action_values))]

        return action

    def add_reward(self, reward):
        self.reward += reward

    def action_to_move(self, action):
        """
        Converting an action (integer between 0 and action_dim) to a move on the board,
        according to the action representation specified in self.piece_action
        :param action: action integer e.g. 3
        :return: move e.g. ((0, 0), (0, 1))
        """
        if action is None:
            return None
        i = self.actions.index(action)
        piece = self.actors[i]
        piece_pos = piece.position  # where is the piece
        if piece_pos is None:
            move = (None, None)  # return illegal move
            return move

        pos_to = (piece_pos[0] + action[0], piece_pos[1] + action[1])
        move = (piece_pos, pos_to)
        return move

    @staticmethod
    def check(piece, team, type_, version):
        return 1 * (piece.team == team and piece.type == type_ and piece.version == version)

    def board_to_state(self, board=None):
        """
        Converts the board of pieces (aka self.board) to the state input for a neural network,
        according to the environment for which the agent is specified
        (e.g. Finder only needs his own position, but MiniStratego will need all opponents pieces also)
        :return: (state_dim * 5 * 5) Tensor
        """

        if board is None:
            board = self.board

        conditions = self.state_represent()
        state_dim = len(conditions)
        board_state = np.zeros((1, state_dim, self.board.shape[0], self.board.shape[1]))  # zeros for empty field
        for pos, val in np.ndenumerate(board):
            p = board[pos]
            if p is not None:  # piece on this field
                for i, (team, type_, vers) in enumerate(conditions):
                    board_state[(0, i) + pos] = self.check(p, team, type_, vers)  # represent type
        board_state = torch.Tensor(board_state).to(GLOBAL_DEVICE.device)
        # add dim for batches
        board_state = board_state.view(1, state_dim, self.board.shape[0], self.board.shape[0])
        return board_state

    def update_prob_by_fight(self, enemy_piece):
        """
        update the information about the given piece, after a fight occured
        :param enemy_piece: object of class Piece
        :return: change is in-place, no value specified
        """
        enemy_piece.potential_types = [enemy_piece.type]

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
            moving_piece.potential_types = [moving_piece.type]  # piece is 2
        else:
            immobile_enemy_types = [idx for idx, type_ in enumerate(moving_piece.potential_types)
                                    if type_ in [0, 11]]
            moving_piece.potential_types = np.delete(moving_piece.potential_types, immobile_enemy_types)

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
        types_alive = [piece.type for piece in enemy_pieces_alive]

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
            piece.type = sample[idx]
            if piece.type in [0, 11]:
                piece.can_move = False
                piece.move_radius = 0
            elif piece.type == 2:
                piece.can_move = True
                piece.move_radius = float('inf')
            else:
                piece.can_move = True
                piece.move_radius = 1
            piece.hidden = False
            board[piece.position] = piece
        return board


class AlphaZero(Reinforce):
    def __init__(self, team, game_dim=5, low_train=False):
        super(AlphaZero, self).__init__(team=team)
        self.canonical_teams = True
        self.invert_moves = bool(team)
        self.low_train = low_train
        self.action_dim = 64  # all pieces 3 * 16 (for pieces: 2, 2, 2) + 4 * 4 for (for pieces 1, 3, 3, 10)
        self.state_dim = len(self.state_represent())

        filter_amounts = np.array([128, 128, 128, 128])
        maxpool_layer_pos = np.array([0, 1, 0, 0])
        width, height = game_dim, game_dim
        for pos in maxpool_layer_pos:
            if pos == 1:
                width = game_dim // 2
                height = width
        d_in = filter_amounts[-1] * width * height
        d_out = self.action_dim
        nr_lin_layers = 5
        kernel_sizes = np.array([3, 5, 3, 5])
        dropout_prob_per_layer = np.array([0.5, 0.5, 0, 0])
        start_layer_exponent = 10
        activation_function = torch.nn.ReLU()
        nnet = models.ELaborateConvFC(game_dim=game_dim,
                                      channels_in=self.state_dim,
                                      filter_amounts=filter_amounts,
                                      maxpool_layer_pos=maxpool_layer_pos,
                                      d_in=d_in,
                                      d_out=d_out,
                                      kernel_sizes=kernel_sizes,
                                      nr_lin_layers=nr_lin_layers,
                                      dropout_prob_per_layer=dropout_prob_per_layer,
                                      start_layer_exponent=start_layer_exponent,
                                      activation_function=activation_function)
        self.model = models.NNetWrapper(game_dim=game_dim, nnet=nnet, action_dim=self.action_dim)
        # self.model = models.Linear(self.state_dim, self.action_dim)
        # self.model.load_state_dict(torch.load('./saved_models/stratego_best.pkl'))

    def decide_move(self, *args, **kwargs):
        self.force_canonical(self.team)
        self.model.to_device()
        pred, _ = self.model.predict(self.board_to_state(self.board))

        if self.low_train:
            actions, relation_dict = utils.action_rep.actions, utils.action_rep.act_piece_relation
            actions_mask = utils.get_actions_mask(self.board, 0,
                                                  relation_dict,
                                                  actions)
            pred = actions_mask * pred

        if actions_mask.sum() == 0:
            self.force_canonical(0)
            # no more legal moves -> lost
            return None

        act = np.argmax(pred)
        move = self.action_to_move(act, 0)

        self.force_canonical(0)
        move = self.invert_move(move)

        return move

    def action_to_move(self, action_id, team, **kwargs):
        """
        Converting an action (integer between 0 and action_dim) to a move on the board,
        according to the action representation specified in self.piece_action
        :param action: action integer e.g. 3
        :return: move e.g. ((0, 0), (0, 1))
        """
        if action_id is None:
            return None
        actions = utils.action_rep.actions
        actors = utils.action_rep.actors
        action = actions[action_id]

        piece_desc = actors[action_id]
        piece = self.relate_actor_desc(piece_desc, team)
        piece_pos = piece.position  # where is the piece

        pos_to = (piece_pos[0] + action[0], piece_pos[1] + action[1])
        move = (piece_pos, pos_to)
        return move

    def relate_actor_desc(self, desc, team):
        type_, version = list(map(int, desc.split('_', 1)))
        for piece in self.board.flatten():
            if piece is not None and piece.type == type_ and piece.version == version and piece.team == team:
                wanted_piece = piece
                break

        return wanted_piece

    def invert_move(self, move):
        if self.invert_moves:
            from_, to_ = move
            game_dim = self.board.shape[0]
            return ((game_dim - 1 - from_[0], game_dim - 1 - from_[1]),
                    (game_dim - 1 - to_[0], game_dim - 1 - to_[1]))
        return move

    def force_canonical(self, player):
        """
        Make the given player be team 0.
        :param player: int, the team to convert to
        """
        if player == 0 and self.canonical_teams:
            # player 0 is still team 0
            return
        elif player == 1 and not self.canonical_teams:
            # player 1 has already been made 0 previously
            return
        else:
            # flip team 0 and 1 and note down the change in teams
            self.canonical_teams = not self.canonical_teams
            self.board = np.flip(self.board)
            for pos, piece in np.ndenumerate(self.board):
                # flip all team attributes
                if piece is not None and piece.team != 99:
                    piece.team ^= 1
                    piece.position = pos

    def state_represent(self):
        conditions = []

        # own team
        # flag, 1 , 10, bombs
        conditions += [(0, t, v) for (t, v) in zip([0, 1, 10, 11], [1]*4)]
        # 2's, 3 versions
        conditions += [(0, t, v) for (t, v) in zip([2]*3, [1, 2, 3])]
        # 3's, 2 versions
        conditions += [(0, t, v) for (t, v) in zip([3]*2, [1, 2])]

        # opponent team
        # flag, 1 , 10, bombs
        conditions += [(1, t, v) for (t, v) in zip([0, 1, 10, 11], [1]*4)]
        # 2's, 3 versions
        conditions += [(1, t, v) for (t, v) in zip([2]*3, [1, 2, 3])]
        # 3's, 2 versions
        conditions += [(1, t, v) for (t, v) in zip([3]*2, [1, 2])]

        # obstacle
        conditions += [(99, 99, 1)]

        return conditions


class OmniscientStratego(AlphaZero):
    def __init__(self, team):
        super().__init__(team)

    def decide_move(self):
        state = self.board_to_state()
        action = self.select_action(state)
        if action is not None:
            move = self.action_to_move(action)
        else:
            return None
        return move


class MiniMax(Agent):
    """
    Agent deciding his moves based on the minimax algorithm. The agent guessed the enemies setup
    before making a decision by using the current information available about the pieces.
    """
    def __init__(self, team, setup=None, depth=None):
        super(MiniMax, self).__init__(team=team, setup=setup)
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

    def decide_move(self):
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
        self.winGameReward = max(self.winGameReward, self.max_depth*self.kill_reward)
        return self.minimax(max_depth=self.max_depth)

    def set_max_depth(self):
        n_alive_enemies = sum([True for piece in self.ordered_opp_pieces if not piece.dead])
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
        :return: tuple of position tuples
        """
        curr_board = copy.deepcopy(self.board)
        curr_board = self.draw_consistent_enemy_setup(curr_board)
        chosen_action = self.max_val(curr_board, 0, -float("inf"), float("inf"), max_depth)[1]
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

        val = -float('inf')
        best_action = None
        for action in my_doable_actions:
            board, fight_result = self.do_move(action, board=board, bookkeeping=False, true_gameplay=False)
            temp_reward = current_reward + self.add_temp_reward(fight_result)
            new_val = self.min_val(board, temp_reward, alpha, beta, depth-1)[0]
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

        val = float('inf')  # initial value set, so min comparison later possible
        best_action = None
        # iterate through all actions
        for action in my_doable_actions:
            board, fight_result = self.do_move(action, board=board, bookkeeping=False, true_gameplay=False)
            temp_reward = current_reward - self.add_temp_reward(fight_result)
            new_val = self.max_val(board, temp_reward, alpha, beta, depth-1)[0]
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
            if piece is not None and piece.type == 0:
                flag_alive[piece.team] = True
        if not flag_alive[self.other_team]:
            return True, True
        # if not flag_alive[self.team]:
        #     return True, False
        if not actions_possible:
            # print('cannot move anymore')
            if max_val:  # the minmax agent is the one doing max_val, so if he cant move -> loss for him
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
            elif not won:
                terminal_reward = - self.winGameReward
            return terminal_reward * (depth + 1) / (self.max_depth + 1) * (terminal_reward / self.kill_reward)

    def update_prob_by_fight(self, enemy_piece):
        """
        update the information about the given piece, after a fight occured
        :param enemy_piece: object of class Piece
        :return: change is in-place, no value specified
        """
        enemy_piece.potential_types = [enemy_piece.type]

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
            moving_piece.potential_types = [moving_piece.type]  # piece is 2
        else:
            immobile_enemy_types = [idx for idx, type in enumerate(moving_piece.potential_types)
                                    if type in [0, 11]]
            moving_piece.potential_types = np.delete(moving_piece.potential_types, immobile_enemy_types)

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
        types_alive = [piece.type for piece in enemy_pieces_alive]

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
            piece.type = sample[idx]
            if piece.type in [0, 11]:
                piece.can_move = False
                piece.move_radius = 0
            elif piece.type == 2:
                piece.can_move = True
                piece.move_radius = float('inf')
            else:
                piece.can_move = True
                piece.move_radius = 1
            piece.hidden = False
            board[piece.position] = piece
        return board

    def undo_last_move(self, board):
        """
        Undo the last move in the memory. Return the updated board.
        :param board: numpy array (5, 5)
        :return: board
        """
        last_move = self.last_N_moves.pop()
        if last_move is None:
            raise ValueError("No last move to undo detected!")
        before_piece = self.pieces_last_N_Moves_beforePos.pop()
        board[last_move[0]] = before_piece
        # the piece at the 'before' position was the one that moved, so needs its
        # last entry in the move history deleted
        before_piece.position = last_move[0]
        board[last_move[1]] = self.pieces_last_N_Moves_afterPos.pop()
        return board


class Omniscient(MiniMax):
    """
    Child of MiniMax agent. This agent is omniscient and thus knows the location and type of each
    piece of the enemy. It then plans by doing a minimax algorithm.
    """
    def __init__(self, team, setup=None, depth=None):
        super(Omniscient, self).__init__(team=team, setup=setup, depth=depth)

    def install_board(self, board, reset=False):
        super().install_board(board, reset)
        self.unhide_all()

    def unhide_all(self):
        """
        Uncover all enemy pieces by setting the hidden attribute to False
        :return: None
        """
        for pos, piece in np.ndenumerate(self.board):
            if piece is not None:
                piece.hidden = False

    def minimax(self, max_depth):
        chosen_action = self.max_val(self.board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action

    def update_prob_by_fight(self, enemy_piece):
        pass

    def update_prob_by_move(self, move, moving_piece):
        pass


class OmniscientHeuristic(Omniscient):
    """
    Omniscient Minimax planner, that uses a learned board heuristic as evaluation function.
    """
    def __init__(self, team, setup=None):
        super(OmniscientHeuristic, self).__init__(team=team, setup=setup)
        self.evaluator = Stratego(team)

    def install_board(self, board, reset=False):
        super().install_board(board, reset)
        self.evaluator.install_board(board, reset)
        self.unhide_all()  # use if inheriting from Omniscient

    def get_network_reward(self):
        state = self.evaluator.board_to_state()
        self.evaluator.model.eval()
        state_action_values = self.evaluator.model(torch.tensor(state)).data.numpy()
        return np.max(state_action_values)

    def get_terminal_reward(self, done, won, depth):
        if not done:
            return self.get_network_reward()
        else:
            if won:
                terminal_reward = self.winGameReward
            elif not won:
                terminal_reward = - self.winGameReward
            return terminal_reward * (depth + 1) / (self.max_depth + 1) * (terminal_reward / self.kill_reward)


class Heuristic(MiniMax):
    """
    Non omniscient Minimax planner with learned board evluation function.
    """
    def __init__(self, team, setup=None):
        super(Heuristic, self).__init__(team=team, setup=setup)
        self.evaluator = Stratego(team)

    def install_board(self, board, reset=False):
        super().install_board(board, reset)
        self.evaluator.install_board(board, reset)

    def get_network_reward(self):
        state = self.evaluator.board_to_state()
        self.evaluator.model.eval()
        state_action_values = self.evaluator.model(torch.tensor(state)).data.numpy()
        return np.max(state_action_values)

    def get_terminal_reward(self, done, won, depth):
        if not done:
            return self.get_network_reward()
        else:
            if won:
                terminal_reward = self.winGameReward
            elif not won:
                terminal_reward = - self.winGameReward
            return terminal_reward * (depth + 1) / (self.max_depth + 1) * (terminal_reward / self.kill_reward)


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
        :return: tuple of position tuples
        """
        possible_moves = utils.get_poss_moves(self.board, self.team)
        next_action = None
        if possible_moves:
            values_of_moves = dict.fromkeys(possible_moves, 0)
            for move in possible_moves:
                for draw in range(self._nr_of_enemy_setups_to_draw):
                    curr_board = self.draw_consistent_enemy_setup(copy.deepcopy(self.board))
                    curr_board, _ = self.do_move(move, curr_board, bookkeeping=False, true_gameplay=False)
                    values_of_moves[move] += self.approximate_value_of_board(curr_board) / self._nr_of_enemy_setups_to_draw
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
                    move = random.choice(actions)
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
                elif turn > self._nr_of_max_turn_sim:  # check if we reached the max number of turns
                    # calculate bonus
                    my_team = self.get_team_from_board(board, self.team)
                    enemy_team = self.get_team_from_board(board, self.other_team)
                    bonus = (len(my_team) - len(enemy_team)) / 20
                    # -1+2*won equals -1+2*0=-1 for won=False, and -1+2*1=1 for won=True
                    # bonus is negative if enemy team has more pieces
                    evals.append(bonus)
                    finished = True
                turn = (turn + 1) % 2
        return sum(evals)/len(evals)

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
            if piece is not None and piece.type == 0:
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
        super(MonteCarloHeuristic, self).__init__(team=team,
                                                  setup=setup,
                                                  number_of_iterations_game_sim=1)
        self.evaluator = Stratego(team)

    def get_network_reward(self):
        state = self.evaluator.board_to_state()
        self.evaluator.model.eval()
        state_action_values = self.evaluator.model(torch.tensor(state)).data.numpy()
        return np.max(state_action_values)

    def decide_move(self):
        """
        given the maximum depth, copy the known board so far, assign the pieces by random, while still
        respecting the current knowledge, and then decide the move via minimax algorithm.
        :return: tuple of position tuples
        """
        possible_moves = utils.get_poss_moves(self.board, self.team)
        next_action = None
        if possible_moves:
            values_of_moves = dict.fromkeys(possible_moves, 0)
            for move in possible_moves:
                for draw in range(self._nr_of_enemy_setups_to_draw):
                    curr_board = self.draw_consistent_enemy_setup(copy.deepcopy(self.board))
                    curr_board, _ = self.do_move(move, curr_board, bookkeeping=False, true_gameplay=False)
                    self.evaluator.board = curr_board
                    values_of_moves[move] += self.get_network_reward() / self._nr_of_enemy_setups_to_draw
                    self.undo_last_move(curr_board)
            evaluations = list(values_of_moves.values())
            actions = list(values_of_moves.keys())
            next_action = actions[evaluations.index(max(evaluations))]
        return next_action

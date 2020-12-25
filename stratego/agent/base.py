import numpy as np
import copy

# from collections import Counter
from scipy import spatial

# from scipy import optimize

import torch
from collections import Counter
import abc


class Agent:
    """
    Agent decides which action to take
    """

    def __init__(self, team):
        self.team = team
        self.other_team = (self.team + 1) % 2

    def decide_move(self, state, logic, *args, **kwargs):
        """
        Implementation of the agent's move for the current round
        :return: tuple of "from" spatial tuple to "to" spatial tuple representing the move
        """
        raise NotImplementedError



class RLAgent(Agent, abc.ABC):
    """
    Agent approximating action-value functions with an artificial neural network
    trained with Q-learning
    """

    def __init__(self, team, setup=None):
        super(RLAgent, self).__init__(team=team, setup=setup)
        self.state_dim = None
        self.action_dim = None
        self.model = None
        self.actors = None
        self.actions = None
        self.relation_dict = None
        self.reward = 0

    def decide_move(self, state):
        board = self.draw_consistent_enemy_setup(copy.deepcopy(self.board))
        state = self.state_to_tensor(board)
        action = self.select_action(state)
        if action is not None:
            move = self.action_to_move(action)
        else:
            return None
        return move

    def state_representation(self, player):
        """
        Specify the state representation as input for the network

        Parameters
        ----------
        player: int,
            the player from whose view the state is to be represented
        """
        return NotImplementedError

    def set_action_rep(self, actors, actions, relation_dict):
        """
        Install the action representation given by the game.
        """
        self.actors = []
        for actor, count in Counter(actors).items():
            type_, version = list(map(int, actor.split("_")))
            for piece in self.board.flatten():
                if (
                    piece is not None
                    and piece.type == type_
                    and piece.version == version
                ):
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
    def check(piece, team, type_, version, hidden):
        if team == 0:
            if not hidden:
                # if it's about team 0, the 'hidden' status is unimportant
                return 1 * (
                    piece.team == team
                    and piece.type == type_
                    and piece.version == version
                )
            else:
                # hidden is only important for the single layer that checks for
                # only this quality!
                return 1 * (piece.team == team and piece.hidden == hidden)

        elif team == 1:
            # for team 1 we only get the info about type and version if it isn't hidden
            # otherwise it will fall into the 'hidden' layer
            if not hidden:
                if piece.hidden:
                    return 0
                else:
                    return 1 * (
                        piece.team == team
                        and piece.type == type_
                        and piece.version == version
                    )
            else:
                return 1 * (piece.team == team and piece.hidden)
        else:
            # only obstace should reach here
            return 1 * (piece.team == team)

    def state_to_tensor(self, state: State):
        """
        Converts the state to the tensor input for a neural network,
        according to the environment for which the agent is specified
        :return: torch.Tensor
        """
        raise NotImplementedError

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
                piece.move_radius = float("inf")
            else:
                piece.can_move = True
                piece.move_radius = 1
            piece.hidden = False
            board[piece.position] = piece
        return board


class OmniscientStratego(AlphaZero):
    def __init__(self, team):
        super().__init__(team)

    def decide_move(self):
        state = self.state_to_tensor()
        action = self.select_action(state)
        if action is not None:
            move = self.action_to_move(action)
        else:
            return None
        return move


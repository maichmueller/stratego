import time
import math
import random
from copy import deepcopy
from cythonized import helpers
import numpy as np

EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, cpuct=1, num_mcts_sims=100):
        self.game = game
        self.nnet = nnet
        self.cpuct = cpuct
        self.num_mcts_sims = num_mcts_sims
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.is_terminal ended for board s
        self.Vs = {}        # stores game.get_poss_moves for board s

    def get_action_prob(self, board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        board.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.num_mcts_sims):
            self.search(board)

        s = str(self.game)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in self.game.get_action_rep().values()]

        if temp==0:
            best_act = np.argmax(counts)
            probs = [0]*len(counts)
            probs[best_act]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current board
        """

        s = str(self.game)

        if s not in self.Es:
            self.Es[s] = state.is_terminal()
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(state.board)
            actions_mask = helpers.get_act_repr_mask(state.board, self.game.move_count % 2,
                                                     self.game.action_rep_dict,
                                                     self.game.action_rep_moves,
                                                     self.game.action_rep_pieces)

            self.Ps[s] = self.Ps[s] * actions_mask  # masking invalid moves
            sum_ps = np.sum(self.Ps[s])
            if sum_ps > 0:
                self.Ps[s] /= sum_ps   # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + actions_mask
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = actions_mask
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        action_rep_dict, action_rep_moves, action_rep_pieces = self.game.get_action_rep()
        for idx, a in enumerate(action_rep_moves):
            if valids[idx]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s,a)])
                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_state = deepcopy(state)
        next_state.do_move(move=a)

        v = self.search(next_state)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v
import math
from copy import deepcopy
import torch

import utils

import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet, cpuct=4, num_mcts_sims=100):
        self.nnet = nnet
        self.cpuct = cpuct
        self.num_mcts_sims = max(1, num_mcts_sims)
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.is_terminal ended for board s
        self.Vs = {}        # stores game.get_poss_moves for board s

    def get_action_prob(self, state, player, expl_rate=1):
        """
        This function performs num_mcts_sims simulations of MCTS starting from
        board.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./expl_rate)
        """
        for i in range(self.num_mcts_sims):
            # print('\rIteration:', i, end='')
            r = self.search(deepcopy(state), player, root=True)

        state.force_canonical(player)
        s = str(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(utils.action_rep.action_dim)]
        # self.game.state.force_canonical(0)  # reset board to natural teams (0 is 0 again)
        if sum(counts) == 0:
            # game ended, thus state doesnt have prob values to choose move from
            return r

        if expl_rate == 0:
            best_act = int(np.argmax(counts))
            probs = [0]*len(counts)
            probs[best_act] = 1
            return probs

        counts = [x**(1./expl_rate) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, state, player, root=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current board
        """
        state.force_canonical(player)
        # get string representation of state
        s = str(state)

        if s not in self.Es:
            self.Es[s] = state.is_terminal()
        elif state.move_count > state.max_nr_turns:
            return 0
        if self.Es[s] != 404:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            # print('Leaf node reached.')
            Ps, v = self.nnet.predict(torch.Tensor(state.state_represent(0)))
            actions, relation_dict = utils.action_rep.actions, utils.action_rep.act_piece_relation
            actions_mask = utils.get_actions_mask(state.board, 0,
                                                  relation_dict,
                                                  actions)
            self.Ps[s] = Ps * actions_mask  # masking invalid moves
            self.Ps[s] /= np.sum(self.Ps[s])  # renormalize
            # else:
            #     # if all valid moves were masked make all valid moves equally probable
            #
            #     # NB! All valid moves may be masked if either your NNet architecture is
            #     # insufficient or you've get overfitting or something else.
            #     # If you have got dozens or hundreds of these messages you
            #     # should pay attention to your NNet and/or training process.
            #     print("All valid moves were masked, do workaround.")
            #     self.Ps[s] = self.Ps[s] + actions_mask
            #     self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = actions_mask
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        # valids = actions_mask
        cur_best = -float('inf')
        best_action = -1

        Ps = self.Ps[s]
        if root:
            dirich_noise = np.random.dirichlet([0.5] * utils.action_rep.action_dim)
            Ps = ((1 - 0.25) * Ps + 0.25 * dirich_noise) * valids
            Ps /= Ps.sum()

        # pick the action with the highest upper confidence bound
        for a in np.where(valids)[0]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * Ps[a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * Ps[a] * math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_action = a

        a = best_action
        move = state.action_to_move(a, 0, force=True)

        state.do_move(move)

        v = self.search(state, (player + 1) % 2, root=False)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
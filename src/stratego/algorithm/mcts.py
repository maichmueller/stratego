from ..agent import RLAgent
from ..core import ActionMap, Logic, Status, Team, State

import math
from copy import deepcopy
from typing import Sequence, Tuple, Dict, List, Optional

import torch

import numpy as np

EPS = 1e-8


class MCTS:
    """
    This class handles the MCTS algorithm.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        action_map: ActionMap,
        logic: Logic,
        cpuct: float = 4.0,
        n_mcts_sims: int = 100,
    ):
        self.network = network
        self.action_map = action_map
        self.logic = logic
        self.cpuct = cpuct
        self.n_mcts_sims = max(1, n_mcts_sims)

        self.Qsa: Dict[Tuple[str, int], float] = {}  # stores Q values for (s, a)
        self.Nsa: Dict[
            Tuple[str, int], int
        ] = {}  # stores #times edge (s, a) was visited
        self.Ns: Dict[str, float] = {}  # stores #times board s was visited
        self.Ps: Dict[str, np.ndarray] = {}  # stores policy (returned by neural net)

        self.Es: Dict[str, Status] = {}  # stores game end status for state s
        self.Vs: Dict[str, np.ndarray] = {}  # stores valid moves for state s

    def policy(
        self,
        state: State,
        agent: RLAgent,
        perspective: Optional[Team] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        This function performs n_mcts_sims simulations of MCTS starting from the given
        state to compute the policy for it.

        Parameters
        ----------
        state: State,
            the state whose policy we want.
        agent: RLAgent,
            the reinforcement learning agent for which the policy is intended.
        perspective: Team,
            the team for which the policy is supposed to be computed.
        temperature: float,
            the exploration temperature T. Higher values will push the policy
            towards a uniform distribution via

            ::math: policy ** (1 / T)

            A value of 1 does not change the policy.

        Returns
        -------
        np.ndarray,
            a policy vector where the probability of the a-th action is proportional to
                Nsa[(s,a)] ** (1 / temperature)
        """
        if perspective is None:
            perspective = agent.team

        for i in range(self.n_mcts_sims):
            value = self.search(deepcopy(state), agent, perspective=perspective)
            assert value != float("inf"), "Computed state value is infinite."

        if perspective != Team.blue and not state.flipped_teams:
            # ensure the chosen perspective is seen as team blue
            state.flip_teams()

        s = str(state)

        counts = np.array(
            [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in self.action_map]
        )

        if state.flipped_teams:
            state.flip_teams()

            # reset the flipping

        if temperature == 0:
            best_act = int(np.argmax(counts))
            policy = np.zeros(len(counts))
            policy[best_act] = 1
            return policy

        counts = counts ** (1.0 / temperature)
        policy = counts / counts.sum()
        return policy

    def search(
        self, state: State, agent: RLAgent, perspective: Team, logic: Logic = Logic()
    ):
        """
        This function performs one iteration of MCTS. It iterates, until a leaf node is found.
        The action chosen at each node is one that has the maximum upper confidence bound as
        in the paper.
        Once a leaf node is found, the neural network is called to return an initial
        policy P and a value value for the state. This value is propagated up the search path.
        In case the leaf node is a terminal state, the outcome is propagated up the search path.
        The values of Ns, Nsa, Qsa are updated.

        Returns
        -------
        float,
            the board value for the agent.
        """
        turn_counter_pre = state.turn_counter

        # (state, action) -> value sign
        sa_to_sign = dict()

        # this simply initializes the variable.
        # If one finds this value later in the tree, then there is a bug in the logic.
        value = float("inf")
        # the first iteration is always the root
        root = True

        while True:
            if state.active_team == Team.red:
                # the network is trained only from the perspective of team blue
                state.flip_teams()
            # get string representation of state
            s = str(state)

            if (state.active_team == perspective) == state.flipped_teams:
                # adjust for the correct perspective:
                # The value needs to be always seen from the perspective of the 'agent'.

                # The condition is logically equivalent to:
                #       (selected team == active player AND teams flipped)
                #    OR (selected team != active player AND teams not flipped)
                # -> Opponent perspective.
                # and in these cases we then need to multiply with -1
                # (assuming symmetric rewards).
                value_sign = -1
            else:
                value_sign = 1

            if s not in self.Es:
                self.Es[s] = logic.get_status(state)

            if self.Es[s] != Status.ongoing:
                # terminal node
                value = self.Es[s].value
                break
            elif s not in self.Ps:
                # leaf node
                value = self._fill_leaf_node(state, s, agent)
                break
            else:
                # has not reached a leaf or terminal node yet, so keep searching
                # by playing according to the current policy
                valids = self.Vs[s]
                policy = self.Ps[s]
                if root:
                    policy = self._make_policy_noisy(policy, valids)
                    # the root is only the first iteration. This information was used
                    # only to add noise to the policy. So now we can deactivate this.
                    root = False

                a = self._select_action(s, policy, valids)
                sa_to_sign[(s, a)] = value_sign

                move = self.action_map.action_to_move(a, state, Team.blue)
                self.logic.execute_move(state, move)

        for (s, a), per in sa_to_sign:
            # for every (state, action) pair: update its Q-value and visitation counter.
            self._update_qsa(s, a, value * per)
            # increment the visitation counter of this state
            self.Ns[s] += 1

        # adjust for team perspective and return the value
        return value * value_sign

    def search_recursive(
        self, state: State, agent: RLAgent, root: bool = False, logic: Logic = Logic()
    ):
        """
        This function performs one iteration of MCTS. It is recursively called
        until a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value value for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Notes
        -----
        Since the board value is computed after game termination during the next
        recursive step, which includes a player-view shift, the returned value is
        always from the perspective of the opponent.

        Returns
        -------
        float,
            the (opponent's) board value for the player.
        """
        if state.active_team == Team.red:
            # the network is trained only from the perspective of team blue
            state.flip_teams()
        # get string representation of state
        s = str(state)

        if s not in self.Es:
            self.Es[s] = logic.get_status(state)
        if self.Es[s] != Status.ongoing:
            # terminal node
            return -self.Es[s].value

        if s not in self.Ps:
            # leaf node
            return -self._fill_leaf_node(state, s, agent)

        valids = self.Vs[s]
        policy = self.Ps[s]
        if root:
            policy = self._make_policy_noisy(policy, valids)

        a = self._select_action(s, policy, valids)
        move = self.action_map.action_to_move(a, state, Team.blue)

        self.logic.execute_move(state, move)

        value = self.search_recursive(state, agent, root=False)

        self._update_qsa(s, a, value)
        self.Ns[s] += 1

        return -value

    def _update_qsa(self, s: str, a: int, value: float):
        s_a = (s, a)
        if s_a in self.Qsa:
            self.Qsa[s_a] = (self.Nsa[s_a] * self.Qsa[s_a] + value) / (
                self.Nsa[s_a] + 1
            )
            self.Nsa[s_a] += 1

        else:
            self.Qsa[s_a] = value
            self.Nsa[s_a] = 1

    def _select_action(
        self, s: str, policy: Sequence[float], valid_actions: Sequence[bool]
    ):
        # pick the action with the highest upper confidence bound
        cur_best = -float("inf")
        best_action = -1
        for a in np.where(valid_actions)[0]:
            a = self.action_map[a]
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * policy[a] * math.sqrt(
                    self.Ns[s]
                ) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * policy[a] * math.sqrt(self.Ns[s] + EPS)

            if u > cur_best:
                cur_best = u
                best_action = a
        return best_action

    def _fill_leaf_node(self, state: State, s: str, agent: RLAgent):
        policy, value = self.network.predict(
            agent.state_to_tensor(state, perspective=Team.blue)
        )
        actions_mask = self.action_map.actions_mask(state.board, agent.team, self.logic)
        policy = policy * actions_mask  # masking invalid moves
        self.Ps[s] = policy / policy.sum()  # normalize to get probabilities
        self.Vs[s] = actions_mask
        self.Ns[s] = 0
        return value

    def _make_policy_noisy(
        self, policy: np.ndarray, valid_actions: np.ndarray, weight: float = 0.25
    ):
        # add noise to the actions to encourage more exploration
        dirichlet_noise = np.random.dirichlet([0.5] * len(self.action_map))
        policy = ((1 - weight) * policy + weight * dirichlet_noise) * valid_actions
        return policy / policy.sum()

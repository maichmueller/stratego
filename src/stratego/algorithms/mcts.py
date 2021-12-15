import math
import timeit
from copy import deepcopy
from functools import singledispatchmethod, partial
from typing import Sequence, Tuple, Dict, List, Optional, Callable, Union, Type

import numpy as np

from stratego.learning import Representation
from stratego.agent import RLAgent
from stratego.core import Action, ActionMap, Logic, Status, Team, State, GameConfig
from stratego.utils import RNG, rng_from_seed


class MCTS:
    """
    This class provides an implementation of UCT-based MCTS algorithm.
    """

    EPS = 1e-10

    Qsa: Dict[Tuple[str, int], float]  # stores Q values for (s, a)
    Nsa: Dict[Tuple[str, int], int]  # stores #times edge (s, a) was visited
    Ns: Dict[str, float]  # stores #times state s was visited

    Ts: Dict[str, Status]  # stores game end status for state s (terminality)
    Ls: Dict[str, np.ndarray]  # stores legal moves for state s

    def __init__(
        self,
        action_map: ActionMap,
        rng: Optional[RNG] = rng_from_seed(),
        cuct: float = 4.0,
        rollout_policy: Optional[Callable[[State, List[Action], int], Action]] = None,
        rollout_heuristic: Optional[Callable[[State, Team], float]] = None,
        rollout_depth: int = float("inf"),
        rollout_timeout: float = float("inf"),
        logic: Logic = Logic(),
    ):
        """

        Parameters
        ----------
        action_map: ActionMap,
            the mapping of indices to actions.
        cuct: float,
            the exploration constant of the UCT algorithm. Common values lie between sqrt(2) to 4. Higher values
            prefer greater exploration in the action selection of MCTS.
        rollout_heuristic: Optional[Callable[[State, Team], float]],
            the heuristic used to evaluate a step if a stop condition was reached, instead of a terminal node.
            It takes in the step and the team from whose perspective the value should be estimated.
            A heuristic needs to be provided, if any of the stop conditions (depth or time) has been set.
        rollout_depth: int,
            The stop condition on the maximum depth to search for in the tree.
        rollout_timeout: float,
            The stop condition to limit the computational time the algorithm can take.
        logic: Logic,
            the underlying Stratego game logic object.
        """

        self.action_map = action_map
        self.rng = rng
        self.cuct: float = cuct

        if rollout_policy is None:

            def _default_rollout_policy(
                rng_: RNG, state: State, action_list: List[Action], depth: int
            ):
                return rng_.choice(action_list)

            rollout_policy = partial(_default_rollout_policy, rng)

        self.rollout_policy = rollout_policy
        self.rollout_depth = rollout_depth
        self.rollout_timeout = rollout_timeout
        if (
            rollout_depth != float("inf") or rollout_timeout != float("inf")
        ) and rollout_heuristic is None:
            raise ValueError("If ")
        self.rollout_heuristic = rollout_heuristic

        self.logic = logic

        self.reset_tree()

    def reset_tree(self):
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ts = {}
        self.Ls = {}

    def policy(
        self,
        state: State,
        agent: RLAgent,
        perspective: Optional[Team] = None,
        nr_sims: int = 1,
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
        nr_sims: int,
            the number of mcts simulations to run.
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

        for i in range(nr_sims):
            value = self.search(deepcopy(state), perspective=perspective)
            assert value != float("inf"), "MCTS estimated state value is infinite."

        s = str(state)

        counts = np.array(
            [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in self.action_map]
        )

        policy = self._apply_temperature(counts, temperature)

        return policy

    def _apply_temperature(self, counts, temperature):
        if temperature == 0:
            best_act = int(np.argmax(counts))
            policy = np.zeros(len(counts))
            policy[best_act] = 1
        else:
            counts = counts ** (1.0 / temperature)
            policy = counts / counts.sum()
        return policy

    def search(self, state: State, perspective: Team):
        """
        This function performs one iteration of MCTS. It iterates, until a leaf node is found.
        The action chosen at each node is one that has the maximum upper confidence bound as
        in the paper.
        Once a leaf node is found, the neural network is called to return an initial
        policy P and a value for the state. This value is propagated up the search path.
        In case the leaf node is a terminal state, the outcome is propagated up the search path.
        The values of Ns, Nsa, Qsa are updated.

        Returns
        -------
        float,
            the board value for the agent.
        """

        # (state, action) -> value sign
        sa_to_sign = dict()

        # depth counter for tree traversal
        depth = 0

        while True:
            # get string representation of state
            s = str(state)

            value_sign = self._get_value_sign(state, perspective)

            if s not in self.Ts:
                self.Ts[s] = self.logic.get_status(state)

            if self.Ts[s] != Status.ongoing:
                # s is a terminal node
                value = self.Ts[s].value
                break  # break out of the while loop
            elif s not in self.Ns:
                # s is a leaf node
                # roll out the game to add the node to the tree, find its value and propagate said value back up
                value = self._expand(state, s, depth, perspective)
                break  # break out of the while loop
            else:
                a = self._select_action(s, depth)
                sa_to_sign[(s, a)] = value_sign
                # apply the move onto the state
                move = self.action_map.action_to_move(
                    a, state.piece_by_id, state.active_team
                )
                self.logic.execute_move(state, move)

                depth += 1

        for (s, a), perspec in sa_to_sign:
            # for every (state, action) pair: update its Q-value and visitation counter.
            self._update_qsa(s, a, value * perspec)
            # increment the visitation counter of this state
            self.Ns[s] += 1

        # adjust for team perspective and return the value
        return value * value_sign

    def _update_qsa(self, s: str, a: int, value: float):
        s_a = (s, a)
        if s_a in self.Qsa:
            # updating the rewards goes by the following steps:
            # 1. Compute the current sum of rewards for action a: Sum_r = avg. rewards (=Qsa) * Nr visits (=Nsa)
            # 2. Add new rewards to it: Sum_r_new = Sum_r + rewards (=value)
            # 3. Average out over new count: avg. rewards new = Sum_r_new / (Nr Visits + 1)
            self.Qsa[s_a] = (self.Nsa[s_a] * self.Qsa[s_a] + value) / (
                self.Nsa[s_a] + 1
            )
            self.Nsa[s_a] += 1
        else:
            self.Qsa[s_a] = value
            self.Nsa[s_a] = 1

    def _select_action(self, s: str, depth: int):
        # pick the action with the highest upper confidence bound
        cur_best = -float("inf")
        best_actions = []
        cuct = self.cuct
        for a in np.where(self.Ls[s])[0]:
            a = self.action_map[a]
            if (s, a) in self.Qsa:
                # compute the value according to UCT: value = avg reward + 2 C sqrt( 2 ln(N_s) / N_sa)
                u = self.Qsa[(s, a)] + 2 * cuct * np.sqrt(
                    2 * np.log(self.Ns[s]) / self.Nsa[(s, a)]
                )
            else:
                # if this action has not been considered before, then give it the highest priority
                u = float("inf")

            if u > cur_best:
                cur_best = u
                best_actions = [a]
            elif math.isclose(u, cur_best, rel_tol=self.EPS):
                best_actions.append(a)

        # if there is more than one best action, select randomly
        return self.rng.choice(best_actions)

    def _expand(
        self, state: State, s: str, depth: int, team: Team,
    ):
        """
        Perform the simulation step of the MCTS algorithm. This will play the game in turn according to the default
        policy and return the value found. If no end has been found before the maximum time or depth ran out, a
        heuristic is asked to evaluate the current state.
        """

        curr_depth = depth
        timer = timeit.default_timer
        runtime = 0.0
        state_copy = deepcopy(state)

        while True:
            start_time = timer()
            # Choose action according to default policy
            action = self.rollout_policy(
                state_copy,
                self.action_map.actions_filtered(
                    state_copy.board, state_copy.active_team, self.logic
                ),
                depth,
            )
            # apply action onto state
            move = self.action_map.action_to_move(
                action, state.piece_by_id, state.active_team
            )
            self.logic.execute_move(state_copy, move)
            # accumulate reward
            if state_copy.status != Status.ongoing:
                if state_copy.status == Status.tie:
                    return Status.tie.value
                else:
                    return Status.win(state_copy.active_team.opponent()).value
            # incr runtime by elapsed amount of time and the depth by 1
            runtime += timer() - start_time
            curr_depth += 1
            # check if any of the configured stop conditions hold.
            if curr_depth == self.rollout_depth or runtime > self.rollout_timeout:
                return self.rollout_heuristic(state_copy, team)

    def _get_value_sign(self, state: State, perspective: Team):
        if state.active_team == perspective:
            return 1
        else:
            return -1


class EvaluatorMCTS(MCTS):
    """
    This class provides an implementation of an evaluator-based MCTS algorithm.
    The differences to the standard MCTS are (let V = value of node, Q = state-action values at node):
        - Instead of a tree rollout, the V, Q of a new node is estimated by an evaluator model (e.g. neural network).

    A usage of this MCTS variant is found in the AlphaZero algorithm.
    """

    Ps: Dict[str, np.ndarray]  # stores policy (returned by evaluator)

    # the evaluator type needs to return a policy and the value for a given state tensor.
    EvaluatorT = Callable[[np.ndarray], Tuple[np.ndarray, float]]

    def __init__(
        self,
        evaluator: EvaluatorT,
        opp_evaluator: EvaluatorT,
        representer: Representation,
        action_map: ActionMap,
        rng: Optional[RNG] = rng_from_seed(),
        eval_needs_flipping: bool = True,
        opp_eval_needs_flipping: bool = True,
        logic: Logic = Logic(),
    ):
        """

        Parameters
        ----------
        evaluator: Callable[[np.ndarray], np.ndarray],
            the main evaluator of states to policies. The states are given as numpy arrays and the returned policy is
            expected to be a numpy array as well.
        opp_evaluator: Callable[[np.ndarray], np.ndarray] or Callable[[np.ndarray], Action],
            In the first case a Callable evaluating a state tensor to a full policy,
            In the second case a Callable evaluating a state tensor to return an action choice.
            For self-play, pass in the same callable as for evaluator.
        representer: Representation,
            the state representation method. Converts state objects to appropriate state tensors.
        action_map: ActionMap,
            the mapping of indices to actions.
        eval_needs_flipping: bool,
            whether the model for evaluating states needs to see itself as team blue (0) for correct evaluation.
        opp_eval_needs_flipping: bool,
            whether the opponent model for evaluating states needs to see itself as team blue (0) for correct evaluation.
        logic: Logic,
            the underlying Stratego game logic object.
        """
        super().__init__(action_map=action_map, rng=rng, logic=logic)
        self.evaluator = evaluator
        self.opp_evaluator = opp_evaluator
        self.current_evaluator = None
        self.representer = representer

        self.eval_flip: bool = eval_needs_flipping
        self.opp_eval_flip: bool = opp_eval_needs_flipping

        self.Ps = {}
        self.reset_tree()

    def reset_tree(self):
        super().reset_tree()
        self.Ps = {}

    def policy(
        self,
        state: State,
        agent: RLAgent,
        perspective: Optional[Team] = None,
        nr_sims: int = 1,
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
        nr_sims: int,
            the number of mcts simulations to run.
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

        for i in range(nr_sims):
            value = self.search(deepcopy(state), perspective=perspective)
            assert value != float("inf"), "MCTS estimated state value is infinite."

        if self.eval_flip and perspective != Team.blue and not state.flipped_teams:
            # ensure the own team is seen as team blue if it is required by the algorithm. This only needs to happen if
            # mcts is called for the red player and the current board setup is not already flipped.
            state.flip_teams()

        s = str(state)

        counts = np.array(
            [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in self.action_map]
        )

        if state.flipped_teams:
            # reset the flipping
            state.flip_teams()

        policy = self._apply_temperature(counts, temperature)

        return policy

    def search(self, state: State, perspective: Team):
        """
        This function performs one iteration of MCTS. It iterates, until a leaf node is found.
        The action chosen at each node is one that has the maximum upper confidence bound as
        in the paper.
        Once a leaf node is found, the neural network is called to return an initial
        policy P and a value for the state. This value is propagated up the search path.
        In case the leaf node is a terminal state, the outcome is propagated up the search path.
        The values of Ns, Nsa, Qsa are updated.

        Returns
        -------
        float,
            the board value for the agent.
        """
        # the evaluator of state nodes will be assigned in each iteration of the step
        self.current_evaluator = None
        return super().search(state, perspective)

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

    def _select_action(self, s: str, depth: int):
        # has not reached a leaf or terminal node yet, so keep searching
        # by playing according to the current policy
        valids = self.Ls[s]
        policy = self.Ps[s]
        if depth == 0:
            policy = self._make_policy_noisy(policy, valids)
            # the root is only the first iteration. This information was used
            # only to add noise to the policy.

        # pick the action with the highest upper confidence bound
        cur_best = -float("inf")
        best_actions = []
        cuct = self.cuct
        for a in np.where(valids)[0]:
            a = self.action_map[a]
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + cuct * policy[a] * np.sqrt(self.Ns[s]) / (
                    1 + self.Nsa[(s, a)]
                )
            else:
                u = cuct * policy[a] * math.sqrt(self.Ns[s] + self.EPS)

            if u > cur_best:
                cur_best = u
                best_actions = [a]
            elif np.isclose(u, cur_best, rel_tol=self.EPS):
                best_actions.append(a)

        # if there is more than one best action, select randomly
        return self.rng.choice(best_actions)

    def _expand(
        self, state: State, s: str, depth: int, team: Team,
    ):
        state_tensor = self.representer(state, own_team=Team.blue)
        if state.active_team == team:
            # it is the mcts-starting player turn
            self.current_evaluator = self.evaluator
            if self.eval_flip and state.active_team != Team.blue:
                # the evaluator needs to see itself as team blue, so we flip
                state.flip_teams()
        else:
            # it is the opponent's turn
            self.current_evaluator = self.opp_evaluator
            if self.opp_eval_flip and state.active_team != Team.blue:
                # the opponent evaluator needs to see itself as team blue, so we flip
                state.flip_teams()

        policy, value = self.current_evaluator(state_tensor)

        if state.flipped_teams:
            # undo the flip
            state.flip_teams()

        actions_mask = self.action_map.actions_mask(state.board, team, self.logic)
        policy = policy * actions_mask  # masking invalid moves
        self.Ps[s] = policy / policy.sum()  # normalize to get probabilities
        self.Ls[s] = actions_mask
        self.Ns[s] = 0
        return value

    def _make_policy_noisy(
        self, policy: np.ndarray, valid_actions: np.ndarray, weight: float = 0.25
    ):
        # add noise to the actions to encourage more exploration
        dirichlet_noise = np.random.dirichlet([0.5] * len(self.action_map))
        policy = ((1 - weight) * policy + weight * dirichlet_noise) * valid_actions
        return policy / policy.sum()

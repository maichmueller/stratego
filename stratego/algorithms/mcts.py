import dataclasses
import math
from copy import deepcopy
from functools import singledispatchmethod, partial
from typing import Sequence, Tuple, Dict, List, Optional, Callable, Union, Type

import torch

import numpy as np

from stratego.learning import Representation
from stratego.agent import RLAgent
from stratego.core import Action, ActionMap, Logic, Status, Team, State
from stratego.utils import RNG, rng_from_seed


class MCTS:
    """
    This class provides an implementation of UCT-based MCTS algorithm.
    """

    EPS = 1e-10

    class Config:
        __slots__ = ["rng", "cuct", "rollout_policy", "rollout_depth"]

        def __init__(
                self,
                rng: Optional[RNG] = rng_from_seed(),
                cuct: float = 4.0,
                rollout_policy: Optional[
                    Callable[[State, List[Action], int], Action]
                ] = None,
                rollout_depth: int = float("inf")
        ):
            """
            Parameters
            ----------
            cuct: float,
                the exploration constant of the UCT algorithm. Common values lie between sqrt(2) to 4. Higher values
                prefer greater exploration in the action selection of MCTS.
            """
            self.rng = rng
            self.cuct: float = cuct

            if rollout_policy is None:
                def _default_rollout_policy(
                        rng: RNG, state: State, action_list: List[Action], depth: int
                ):
                    return rng.choice(action_list)

                rollout_policy = partial(_default_rollout_policy, rng)
            self.rollout_policy = rollout_policy
            self.rollout_depth = rollout_depth

    Qsa: Dict[Tuple[str, int], float]  # stores Q values for (s, a)
    Nsa: Dict[Tuple[str, int], int]  # stores #times edge (s, a) was visited
    Ns: Dict[str, float]  # stores #times state s was visited

    Ts: Dict[str, Status]  # stores game end status for state s (terminality)
    Ls: Dict[str, np.ndarray]  # stores legal moves for state s

    def __init__(
            self, action_map: ActionMap, config: Config = Config(), logic: Logic = Logic(),
    ):
        """

        Parameters
        ----------
        action_map: ActionMap,
            the mapping of indices to actions.
        logic: Logic,
            the underlying Stratego game logic object.
        """
        self.action_map = action_map
        self.logic = logic
        self.config = config

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

    def search(self, state: State, perspective: Team, logic: Logic = Logic()):
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

        # this simply initializes the variable.
        # If one finds this value later in the tree, then there is a bug in the logic.
        value = float("inf")

        # depth counter for tree traversal
        depth = 0

        # the evaluator of state nodes will be assigned in each iteration of the step
        evaluator = None

        while True:

            if state.active_team == Team.red:
                # the network is trained only from the perspective of team blue
                state.flip_teams()

            # get string representation of state
            s = str(state)

            value_sign = self._get_value_sign(state, perspective)

            if s not in self.Ts:
                self.Ts[s] = logic.get_status(state)

            if self.Ts[s] != Status.ongoing:
                # terminal node
                value = self.Ts[s].value
                break  # break out of the while loop
            elif s not in self.Ns:
                # leaf node
                # roll out the game to find the value of this node and backpropagate it
                value = self._expand(state, s, perspective)
                break  # break out of the while loop
            else:
                a = self._select_action(s, depth)
                sa_to_sign[(s, a)] = value_sign
                # apply the move onto the state
                move = self.action_map.action_to_move(a, state.piece_by_id, state.active_team)
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
        cuct = self.config.cuct
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
        return self.config.rng.choice(best_actions)

    def _expand(
            self, state: State, s: str, depth: int, team: Team,
    ):
        reward = 0.
        curr_depth = depth
        state_copy = deepcopy(state)

        for i in range(self.config.rollout_depth):
            # Choose action according to default policy
            action = self.config.rollout_policy(state_copy, self.action_map.actions, depth)
            # apply action onto state
            move = self.action_map.action_to_move(action, state.piece_by_id, state.active_team)
            self.logic.execute_move(state_copy, move)
            # accumulate reward
            if state_copy.status != Status.ongoing:
                if state_copy.status == Status.tie:

                    return Status.win(state_copy.active_team.opponent())
                else:
                    return Status.tie


        // Check if terminal
        state
        if (cur_state->terminal())
        break;
        discount *= _gamma;

    }

    return reward;

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

    EPS = 1e-10

    class Config:
        def __init__(
                self,
                cuct: float = 4.0,
                eval_needs_flipping: bool = True,
                opp_eval_needs_flipping: bool = True,
        ):
            """
            Parameters
            ----------
            cuct: float,
                the exploration constant of the UCT algorithm. Common values lie between sqrt(2) to 4. Higher values
                prefer greater exploration in the action selection of MCTS.
            eval_needs_flipping: bool,
                whether the model for evaluating states needs to see itself as team blue (0) for correct evaluation.
            opp_eval_needs_flipping: bool,
                whether the opponent model for evaluating states needs to see itself as team blue (0) for correct evaluation.
            """
            self.cuct: float = cuct
            self.eval_flip: bool = eval_needs_flipping
            self.opp_eval_flip: bool = opp_eval_needs_flipping

    Qsa: Dict[Tuple[str, int], float]  # stores Q values for (s, a)
    Nsa: Dict[Tuple[str, int], int]  # stores #times edge (s, a) was visited
    Ns: Dict[str, float]  # stores #times state s was visited
    Ps: Dict[str, np.ndarray]  # stores policy (returned by evaluator)

    Ts: Dict[str, Status]  # stores game end status for state s (terminality)
    Ls: Dict[str, np.ndarray]  # stores legal moves for state s

    # the evaluator type needs to return a policy and the value for a given state tensor.
    EvaluatorT = Callable[[np.ndarray], Tuple[np.ndarray, float]]

    def __init__(
            self,
            evaluator: EvaluatorT,
            opp_evaluator: EvaluatorT,
            representer: Representation,
            action_map: ActionMap,
            rng: Optional[RNG] = rng_from_seed(),
            config: Config = Config(),
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
        logic: Logic,
            the underlying Stratego game logic object.
        """
        self.evaluator = evaluator
        self.opp_evaluator = opp_evaluator
        self.current_evaluator = None
        self.representer = representer
        self.action_map = action_map
        self.rng = rng
        self.logic = logic
        self.config = config

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

    def search(self, state: State, perspective: Team, logic: Logic = Logic()):
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

        # this simply initializes the variable.
        # If one finds this value later in the tree, then there is a bug in the logic.
        value = float("inf")

        # the first iteration is always the root
        root = True

        # the evaluator of state nodes will be assigned in each iteration of the step
        self.current_evaluator = None

        while True:

            if state.active_team == Team.red:
                # the network is trained only from the perspective of team blue
                state.flip_teams()

            # get string representation of state
            s = str(state)

            value_sign = self._get_value_sign(state, perspective)

            if s not in self.Ts:
                self.Ts[s] = logic.get_status(state)

            if self.Ts[s] != Status.ongoing:
                # terminal node
                value = self.Ts[s].value
                break  # break out of the while loop
            elif s not in self.Ps:
                # leaf node
                # this step here differs from the standard mcts algorithm. Instead of performing a rollout of the game,
                # which would give us the value of this leaf node, we ask the evaluator (e.g. a neural net) to provide
                # an estimate of the policy and value.
                value = self._expand(state, s, perspective)
                break  # break out of the while loop
            else:
                # has not reached a leaf or terminal node yet, so keep searching
                # by playing according to the current policy
                valids = self.Ls[s]
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

        for (s, a), perspec in sa_to_sign:
            # for every (state, action) pair: update its Q-value and visitation counter.
            self._update_qsa(s, a, value * perspec)
            # increment the visitation counter of this state
            self.Ns[s] += 1

        # adjust for team perspective and return the value
        return value * value_sign

    def _get_value_sign(self, state, perspective):
        if (state.active_team == perspective) == state.flipped_teams:
            # adjust the value for the correct perspective:
            # The value needs to always be seen from the given perspective.

            # The condition is logically equivalent to:
            #       (selected team == active player AND teams flipped)
            #    OR (selected team != active player AND teams not flipped)
            # This implies we are currently in an Opponent node.
            # In these cases we need to multiply the value with -1. We also need to switch the evaluator.
            self.current_evaluator = self.opp_evaluator
            value_sign = -1
        else:
            # we are at a node of the player who initialized the MCTS tree. That team is given by 'perspective'.
            self.current_evaluator = self.evaluator
            value_sign = 1
        return value_sign

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
        best_actions = []
        cuct = self.config.cuct
        for a in np.where(valid_actions)[0]:
            a = self.action_map[a]
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + cuct * policy[a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)]
                )
            else:
                u = cuct * policy[a] * math.sqrt(self.Ns[s] + self.EPS)

            if u > cur_best:
                cur_best = u
                best_actions = [a]
            elif math.isclose(u, cur_best, rel_tol=self.EPS):
                best_actions.append(a)

        # if there is more than one best action, select randomly
        return self.rng.choice(best_actions)

    def _expand(
            self, state: State, s: str, team: Team,
    ):
        state_tensor = self.representer(state, own_team=Team.blue)
        policy_or_action, value = self.current_evaluator(state_tensor)
        if isinstance(policy_or_action, np.ndarray):
            policy = policy_or_action
        elif isinstance(policy_or_action, int):
            policy = np.zeros(len(self.action_map), dtype=np.float)
            policy[policy_or_action] = 1.0
        else:
            raise TypeError(
                "Returned type of policy or action from evaluator has to be a np.ndarray or int."
            )
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

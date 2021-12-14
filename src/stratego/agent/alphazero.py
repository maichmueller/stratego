from stratego.agent import DRLAgent
from stratego.core import Team, State
from stratego.learning import PolicyMode


class AlphaZeroAgent(DRLAgent):
    """
    AlphaZero agent. Estimates policy and value with its model.
    """
    def decide_move(self, state: State, *args, **kwargs):
        if self.team == Team.red:
            state.flip_teams()

        self.model.to(self.device)  # no-op if already on correct device

        state_tensor = self.state_to_tensor(state)
        policy, _ = self.model.predict(state_tensor)

        action = self.sample_action(policy, mode=PolicyMode.greedy)
        if state.flipped_teams:
            state.flip_teams()
            action = self.action_map.invert_action(action)

        move = self.action_map.action_to_move(action, state, self.team)
        return move

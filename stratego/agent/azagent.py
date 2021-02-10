from stratego.agent import RLAgent
from stratego.engine import Team, State


class AZAgent(RLAgent):
    """
    AlphaZero agent. Estimates policy and value with its model.
    """
    def decide_move(self, state: State, *args, **kwargs):
        if self.team == Team.red:
            state.flip_teams()

        self.model.to(self.device)  # no-op if already on correct device

        state_tensor = self.state_to_tensor(state)
        policy, _ = self.model.predict(state_tensor)

        action = self.select_action(policy)
        if state.flipped_teams:
            state.flip_teams()
            action = self.action_map.invert_action(action)

        move = self.action_map.action_to_move(action, state, self.team)
        return move

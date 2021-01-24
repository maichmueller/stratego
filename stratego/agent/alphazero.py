from stratego.engine.action import ActionMap
from stratego.agent import RLAgent
import torch
import numpy as np
from stratego.engine import Team, State, Logic
from stratego.learning import models


class AlphaZero(RLAgent):

    _representation_filters = []

    def __init__(self, team, action_map: ActionMap, logic: Logic = Logic(), device="cpu"):
        super().__init__(team=team, action_map=action_map)
        self.canonical_teams = True
        self.state_dim = len(self.state_representation())
        self.device = device
        self.logic = logic

        filter_amounts = np.array([128, 128, 128, 128])
        maxpool_layer_pos = np.array([0, 1, 0, 0])
        width, height = game_size, game_size
        for pos in maxpool_layer_pos:
            if pos == 1:
                width = game_size // 2
                height = width
        d_in = filter_amounts[-1] * width * height
        d_out = self.action_dim
        nr_lin_layers = 5
        kernel_sizes = np.array([3, 5, 3, 5])
        dropout_prob_per_layer = np.array([0.5, 0.5, 0, 0])
        start_layer_exponent = 10
        activation_function = torch.nn.ReLU()
        nnet = models.ELaborateConvFC(
            game_size=game_size,
            channels_in=self.state_dim,
            filter_amounts=filter_amounts,
            maxpool_layer_pos=maxpool_layer_pos,
            d_in=d_in,
            d_out=d_out,
            kernel_sizes=kernel_sizes,
            nr_lin_layers=nr_lin_layers,
            dropout_prob_per_layer=dropout_prob_per_layer,
            start_layer_exponent=start_layer_exponent,
            activation_function=activation_function,
        )
        self.model = models.NetworkWrapper(
            game_size=game_size, net=nnet, action_dim=self.action_dim
        )
        # self.model = models.Linear(self.state_dim, self.action_dim)
        # self.model.load_state_dict(torch.load('./saved_models/stratego_best.pkl'))

    def decide_move(self, state: State, *args, **kwargs):
        self.force_canonical(self.team)
        self.model.to_device()
        board_state = self.state_to_tensor(state)
        pred, _ = self.model.predict(board_state)

        actions_mask = logic.mask_actions(self.board, 0, relation_dict, actions)
        pred = actions_mask * pred

        if actions_mask.sum() == 0:
            self.force_canonical(0)
            # no more legal moves
            return None

        act = np.argmax(pred)
        move = self.action_to_move(act, 0)

        self.force_canonical(0)
        move = self.invert_move(move)

        return move

    def state_to_tensor(self, state):
        board = state.board
        conditions = self.state_representation(self.team)
        state_dim = len(conditions)
        board_state = np.zeros(
            (1, state_dim, self.board.shape[0], self.board.shape[1])
        )  # zeros for empty field
        for pos, val in np.ndenumerate(board):
            p = board[pos]
            if p is not None:  # piece on this field
                for i, (team, type_, vers, hidden) in enumerate(conditions):
                    board_state[(0, i) + pos] = self.check(
                        p, team, type_, vers, hidden
                    )  # represent type
        board_state = torch.Tensor(board_state).to(GLOBAL_DEVICE.device)
        # add dim for batches
        board_state = board_state.view(
            1, state_dim, self.board.shape[0], self.board.shape[0]
        )
        return board_state

    def choose_action(self, state):
        self.model.eval()
        state_action_values = self.model(state).view(-1)
        action = self.action_map[int(torch.argmax(state_action_values))]
        return action

    def state_representation(self, player: Team):
        conditions = []
        for team in [player, player.opponent()]:
            # flag, 1 , 10, bombs
            conditions += [
                (team, t, v, h) for (t, v, h) in zip([0, 1, 10, 11], [1] * 4, [0] * 4)
            ]
            # 2's, 3 versions
            conditions += [
                (team, t, v, h) for (t, v, h) in zip([2] * 3, [1, 2, 3], [0] * 3)
            ]
            # 3's, 2 versions
            conditions += [
                (team, t, v, h) for (t, v, h) in zip([3] * 2, [1, 2], [0] * 2)
            ]
            # all own hidden pieces
            conditions += [(team, None, None, 1)]

        # obstacle
        conditions += [(99, 99, 1, 1)]

        return conditions

    @staticmethod
    def check(piece, team, type_, version, hidden):
        if team == 0:
            if not hidden:
                # if it's about team 0, the 'hidden' status is unimportant
                return 1 * (piece.team == team and int == type_ and int == version)
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
                    return 1 * (piece.team == team and int == type_ and int == version)
            else:
                return 1 * (piece.team == team and piece.hidden)
        else:
            # only obstacle should reach here
            return 1 * (piece.team == team)


class OmniscientStratego(AlphaZero):
    def __init__(self, team):
        super().__init__(team)

    def decide_move(self):
        state = self.state_to_tensor()
        action = self.choose_action(state)
        if action is not None:
            move = self.action_to_move(action)
        else:
            return None
        return move

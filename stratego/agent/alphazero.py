from stratego.agent import RLAgent
import torch
import numpy as np
from stratego import utils, models
from stratego.utils import GLOBAL_DEVICE


class AlphaZero(RLAgent):

    _representation_filters = []

    def __init__(self, team, game_size=5, low_train=False):
        super(AlphaZero, self).__init__(team=team)
        self.canonical_teams = True
        self.invert_moves = bool(team)
        self.action_dim = 64  # all pieces 3 * 16 (for pieces: 2, 2, 2) + 4 * 4 for (for pieces 1, 3, 3, 10)
        self.state_dim = len(self.state_representation())

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
        self.model = models.NNetWrapper(
            game_size=game_size, nnet=nnet, action_dim=self.action_dim
        )
        # self.model = models.Linear(self.state_dim, self.action_dim)
        # self.model.load_state_dict(torch.load('./saved_models/stratego_best.pkl'))

    def decide_move(self, *args, **kwargs):
        self.force_canonical(self.team)
        self.model.to_device()
        board_state = self.state_to_tensor(self.board)
        pred, _ = self.model.predict(board_state)

        actions, relation_dict = (
            utils.action_rep.actions,
            utils.action_rep.piecetype_to_actionrange,
        )
        actions_mask = utils.get_actions_mask(self.board, 0, relation_dict, actions)
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
        type_, version = list(map(int, desc.split("_", 1)))
        for piece in self.board.flatten():
            if (
                piece is not None
                and int == type_
                and int == version
                and piece.team == team
            ):
                wanted_piece = piece
                break

        return wanted_piece

    def invert_move(self, move):
        if self.invert_moves:
            from_, to_ = move
            game_size = self.board.shape[0]
            return (
                (game_size - 1 - from_[0], game_size - 1 - from_[1]),
                (game_size - 1 - to_[0], game_size - 1 - to_[1]),
            )
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

    def state_representation(self, player):
        conditions = []
        for team in [player, (player + 1) % 2]:
            # flag, 1 , 10, bombs
            conditions += [
                (team, t, v, h) for (t, v, h) in zip([0, 1, 10, 11], [1] * 4, [0] * 4)
            ]
            # 2's, 3 versions
            conditions += [
                (team, t, v, h) for (t, v, h) in zip([2] * 3, [1, 2, 3], [0] * 3)
            ]
            # 3's, 2 versions
            conditions += [(team, t, v, h) for (t, v, h) in zip([3] * 2, [1, 2], [0] * 2)]
            # all own hidden pieces
            conditions += [(team, None, None, 1)]

        # obstacle
        conditions += [(99, 99, 1, 1)]

        return conditions


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

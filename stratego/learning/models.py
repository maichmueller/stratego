from abc import ABC
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
from tqdm import tqdm

from stratego.engine import ActionMap, GameSpecification


class NetworkWrapper(torch.nn.Module, ABC):
    def __init__(
        self,
        action_map: Optional[ActionMap] = None,
        game_size: Optional[int] = None,
        device="cpu",
    ):
        super().__init__()
        self.game_size = game_size
        self.action_map = (
            action_map
            if action_map is not None
            else ActionMap(GameSpecification(game_size))
        )
        self.device = device

    def to(self, device: str = "cpu"):
        self.device = device
        self.net.to(device)

    @torch.no_grad()
    def predict(self, board):
        """
        board: np array with board
        """
        self.eval()
        pi, v = self.net(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        torch.save(
            {
                "state_dict": self.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(
        self, folder="checkpoint", filename="checkpoint.pth.tar", device: str = "cpu"
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])


class NNConvolutional(nn.Module):
    """
    Convenience class to create convolutional layers with optional max pooling and dropout in between
    """

    def __init__(
        self,
        channels_in,
        filter_amounts,
        kernel_sizes=None,
        maxpool_layer_pos=None,
        dropout_prob_per_layer=None,
    ):
        super().__init__()
        self.nr_conv_layers = len(filter_amounts)

        if isinstance(dropout_prob_per_layer, int):
            self.dropout_prob_per_layer = dropout_prob_per_layer * np.ones(
                self.nr_conv_layers
            )
        elif isinstance(dropout_prob_per_layer, np.ndarray):
            self.dropout_prob_per_layer = dropout_prob_per_layer
        elif dropout_prob_per_layer is None:
            self.dropout_prob_per_layer = np.zeros(self.nr_conv_layers)

        if maxpool_layer_pos is None:
            self.maxpool_layer_pos = np.zeros(self.nr_conv_layers)
        else:
            self.maxpool_layer_pos = np.array(maxpool_layer_pos)

        if kernel_sizes is None:
            kernel_sizes = 3 * np.ones(self.nr_conv_layers)
        else:
            # all kernel sizes should be odd numbers
            assert np.sum(kernel_sizes % 2) == self.nr_conv_layers

        # calculate the zero padding for each filter size
        zero_paddings = np.zeros(self.nr_conv_layers, dtype=int)
        for idx, kernel_size in enumerate(kernel_sizes):
            zero_paddings[idx] = (kernel_size - 1) / 2

        self.conv_layers = nn.ModuleList()
        # this conversion needs to happen because of some internal torch problem with numpy.int.32
        filter_amounts = [channels_in] + list(map(int, filter_amounts))

        for k in range(self.nr_conv_layers):
            self.conv_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=filter_amounts[k],
                        out_channels=filter_amounts[k + 1],
                        kernel_size=kernel_sizes[k],
                        padding=zero_paddings[k],
                    ),
                    nn.ReLU(),
                ]
            )
            if self.maxpool_layer_pos[k] == 1:
                self.conv_layers.extend([nn.MaxPool2d(kernel_size=3, stride=2)])
            if self.dropout_prob_per_layer[k] > 0:
                self.conv_layers.extend(
                    [nn.Dropout2d(p=self.dropout_prob_per_layer[k])]
                )

    def forward(self, x):
        x = x.to(GLOBAL_DEVICE.device)
        for layer in self.conv_layers:
            x = layer(x)
        return x


class NNLinear(nn.Module):
    """
    Convenience class to create a chain of linear layers
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        nr_lin_layers: int,
        start_layer_exponent: int = 8,
        activation_function: torch.nn.functional = nn.ReLU(),
        device: str = "cpu",
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.start_layer_exponent = start_layer_exponent
        self.hidden_nodes = 2 ** start_layer_exponent
        self.nr_lin_layers = nr_lin_layers
        self.linear_layers = nn.ModuleList([nn.Linear(self.dim_in, self.hidden_nodes)])
        for i in range(self.nr_lin_layers - 2):
            denom1 = 2 ** i
            denom2 = 2 ** (i + 1)
            self.linear_layers.extend(
                [nn.Linear(self.hidden_nodes / denom1, self.hidden_nodes / denom2)]
            )

        self.linear_layers.extend(
            [nn.Linear(self.hidden_nodes / (2 ** (self.nr_lin_layers - 2)), self.dim_out)]
        )
        self.activation = activation_function
        self.to(device)

    def forward(self, x):
        for lin_layer in self.linear_layers[:-1]:
            x = self.activation(lin_layer(x))
        x = self.linear_layers[-1](x)
        return x


class ELaborateConvFC(nn.Module):
    def __init__(
        self,
        game_size,
        channels_in,
        filter_amounts,
        d_in,
        d_out,
        nr_lin_layers,
        kernel_sizes=None,
        maxpool_layer_pos=None,
        dropout_prob_per_layer=None,
        start_layer_exponent=8,
        activation_function=nn.ReLU(),
    ):
        super().__init__()
        self.conv_net = NNConvolutional(
            channels_in,
            filter_amounts,
            kernel_sizes,
            maxpool_layer_pos,
            dropout_prob_per_layer,
        )
        self.d_in = d_in
        self.fc_net = NNLinear(
            d_in, d_out, nr_lin_layers, start_layer_exponent, activation_function
        )

        out_features = self.fc_net.linear_layers[-1].in_features
        self.fc_net.linear_layers = self.fc_net.linear_layers[:-1]
        self.action_value_layer = nn.Linear(
            in_features=out_features, out_features=self.fc_net.dim_out
        )
        self.board_value_layer = nn.Linear(in_features=out_features, out_features=1)
        self.game_size = game_size

    def extract_features(self, x):
        x = x.to(GLOBAL_DEVICE.device)
        params = self.named_parameters()
        output_per_layer = []
        for layer in self.conv_net.conv_layers:
            x = layer(x)
            output_per_layer.append(x)
        for layer in self.fc_net.linear_layers:
            x = self.fc_net.activation(layer(x))
            output_per_layer.append(x)
        return params, output_per_layer

    def forward(self, x):
        x.to(GLOBAL_DEVICE.device)
        x = self.conv_net(x)
        x = x.view(-1, self.d_in)
        x = self.fc_net(x)

        pi = self.action_value_layer(x)  # batch_size x action_size
        v = self.board_value_layer(x)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

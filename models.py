import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniConv(nn.Module):
    """
    One convolution, one linear layer
    """
    def __init__(self, state_dim, action_dim):
        super(MiniConv, self).__init__()
        self.filter = 10
        self.feature_size = self.filter * 25
        self.conv1 = nn.Conv2d(state_dim, self.filter, padding=1, kernel_size=3)
        self.lin1 = nn.Linear(self.feature_size, 16)
        self.lin_final = nn.Linear(16, action_dim)

    def extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.feature_size)
        return F.relu(self.lin1(x))

    def forward(self, x):
        x = self.extract_features(x)
        return F.sigmoid(self.lin_final(x))


class NNLinear(nn.Module):
    """
    Convenience class to create a chain of linear layers
    """

    def __init__(self, D_in, D_out, nr_lin_layers, start_layer_exponent=8,
                 activation_function=nn.ReLU()):
        super().__init__()
        self.cuda_available = torch.cuda.is_available()
        self.D_in = D_in
        self.start_layer_exponent = start_layer_exponent
        self.Hidden = int(pow(2, start_layer_exponent))
        self.D_out = D_out
        self.nr_lin_layers = nr_lin_layers
        self.linear_layers = nn.ModuleList([nn.Linear(self.D_in, self.Hidden)])
        for i in range(self.nr_lin_layers - 2):
            denom1 = int(pow(2, i))
            denom2 = int(pow(2, i + 1))
            self.linear_layers.extend(
                [nn.Linear(int(self.Hidden / denom1), int(self.Hidden / denom2))])

        self.linear_layers.extend(
            [nn.Linear(int(self.Hidden / (pow(2, self.nr_lin_layers - 2))), self.D_out)])
        self.activation_function = activation_function

    def forward(self, x):
        for i in range(self.nr_lin_layers - 1):
            x = self.activation_function(self.linear_layers[i](x))
        x = self.linear_layers[-1](x)
        return x


class NNConvolutional(nn.Module):
    """
    Convenience class to create convolutional layers with optional max pooling and dropout in between
    """

    def __init__(self, channels_in, filter_amounts, kernel_sizes=None,
                 maxpool_layer_pos=None, dropout_prob_per_layer=None):
        super().__init__()
        self.nr_conv_layers = len(filter_amounts)

        if isinstance(dropout_prob_per_layer, int):
            self.dropout_prob_per_layer = dropout_prob_per_layer * np.ones(self.nr_conv_layers)
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
            assert(np.sum(kernel_sizes % 2) == self.nr_conv_layers)

        # calculate the zero padding for each filter size
        zero_paddings = np.zeros(self.nr_conv_layers)
        for idx, kernel_size in enumerate(kernel_sizes):
            zero_paddings[idx] = (kernel_size-1)/2

        self.cuda_available = torch.cuda.is_available()
        self.conv_layers = nn.ModuleList()
        # this conversion needs to happen because of some internal torch problem with numpy.int.32
        filter_amounts = [channels_in] + [int(x) for x in filter_amounts]

        for k in range(self.nr_conv_layers):
            self.conv_layers.extend([nn.Conv2d(in_channels=filter_amounts[k],
                                               out_channels=filter_amounts[k+1],
                                               kernel_size=kernel_sizes[k],
                                               padding=zero_paddings[k]),
                                     nn.Tanh()])
            if self.maxpool_layer_pos[k] == 1:
                self.conv_layers.extend([nn.MaxPool2d(kernel_size=3, stride=2)])
            if self.dropout_prob_per_layer[k] > 0:
                self.conv_layers.extend([nn.Dropout2d(p=self.dropout_prob_per_layer[k])])

    def forward(self, x):
        if self.cuda_available:
            x = x.data.cuda()
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ELaborateConvFC(nn.Module):
    def __init__(self, game_dim, channels_in, filter_amounts, d_in, d_out, nr_lin_layers,
                 kernel_sizes=None, maxpool_layer_pos=None, dropout_prob_per_layer=None,
                 start_layer_exponent=8,
                 activation_function=nn.ReLU()):
        super().__init__()
        self.conv_net = NNConvolutional(channels_in, filter_amounts,
                 kernel_sizes, maxpool_layer_pos, dropout_prob_per_layer)
        self.d_in = d_in
        self.fully_connected_net = NNLinear(d_in, d_out, nr_lin_layers,
                                            start_layer_exponent, activation_function)
        self.game_dim = game_dim

    def extract_features(self, x):
        params = self.named_parameters()
        output_per_layer = []
        for layer in self.conv_net.conv_layers:
            x = layer(x)
            output_per_layer.append(x)
        for layer in self.fully_connected_net.linear_layers:
            x = layer(x)
            output_per_layer.append(x)
        return params, output_per_layer

    def forward(self, x):
        for layer in self.conv_net.conv_layers:
            x = layer(x)
        x = x.view(-1, self.d_in)
        for layer in self.fully_connected_net.linear_layers:
            x = layer(x)
        return x


class SmallConv(nn.Module):
    """
    One convolution, two linear layers
    """
    def __init__(self, state_dim, action_dim, n_filter, n_hidden):
        super(SmallConv, self).__init__()
        self.feature_size = n_filter * 25
        self.conv1 = nn.Conv2d(state_dim, n_filter, padding=1, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(n_filter)
        self.lin1 = nn.Linear(self.feature_size, n_hidden)
        self.lin_final = nn.Linear(n_hidden, action_dim)

    def extract_features(self, x):
        x = F.tanh(self.conv1_bn(self.conv1(x)))
        x = x.view(-1, self.feature_size)
        return F.relu(self.lin1(x))

    def forward(self, x):
        x = self.extract_features(x)
        return F.sigmoid(self.lin_final(x))


class MoreLayer(nn.Module):
    """
    One convolution, three linear layers
    """
    def __init__(self, state_dim, action_dim, n_filter, n_hidden):
        super(MoreLayer, self).__init__()
        self.feature_size = n_filter * 25
        self.conv1 = nn.Conv2d(state_dim, n_filter, padding=1, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(n_filter)
        self.lin1 = nn.Linear(self.feature_size, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin_final = nn.Linear(n_hidden, action_dim)

    def extract_features(self, x):
        x = F.tanh(self.conv1_bn(self.conv1(x)))
        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        return F.relu(self.lin2(x))

    def forward(self, x):
        x = self.extract_features(x)
        return F.sigmoid(self.lin_final(x))


class FullConnect(nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden=256):
        super(FullConnect, self).__init__()
        self.feature_size = state_dim * 25

        self.lin1 = nn.Linear(self.feature_size, n_hidden)
        self.lin_final = nn.Linear(n_hidden, action_dim)

    def extract_features(self, x):
        x = x.view(-1, self.feature_size)
        return F.relu(self.lin1(x))

    def forward(self, x):
        x = self.extract_features(x)
        x = F.sigmoid(self.lin_final(x))
        return x


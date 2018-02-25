import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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


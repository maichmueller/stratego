import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import os
from cythonized.utils import AverageMeter
from progressBar.progress.bar import Bar


class NNetWrapper:
    def __init__(self, nnet, game_dim, action_dim):
        self.nnet = nnet
        if nnet.device.type != 'cpu':
            nnet.cuda()
        self.board_x, self.board_y = game_dim, game_dim
        self.action_size = action_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, examples, epochs, batch_size=128):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(epochs):
            print('\rEPOCH ::: ' + str(epoch+1), end='')
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples) / batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples) / batch_size):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                try:
                    boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                except Exception as e:
                    import re
                    error_pos = int(re.search('(?<=#)\d+\s', 'zip argument #7 not').group())
                    print(*examples[error_pos], sep='\n')
                    raise e
                boards = torch.Tensor(np.array(boards).astype(np.float64))
                target_pis = torch.Tensor(np.array(pis))
                target_vs = torch.Tensor(np.array(vs).astype(np.float64))

                # predict
                # boards, target_pis, target_vs = list(map(lambda x: x.contiguous().to(self.device),
                #                                          [boards, target_pis, target_vs]))

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} ' \
                             '| ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                              batch=batch_idx,
                              size=int(len(examples) / batch_size),
                              data=data_time.avg,
                              bt=batch_time.avg,
                              total=bar.elapsed_td,
                              eta=bar.eta_td,
                              lpi=pi_losses.avg,
                              lv=v_losses.avg
                             )
                bar.next()
            bar.finish()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if self.device != 'cpu' else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


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
            assert (np.sum(kernel_sizes % 2) == self.nr_conv_layers)

        # calculate the zero padding for each filter size
        zero_paddings = np.zeros(self.nr_conv_layers, dtype=int)
        for idx, kernel_size in enumerate(kernel_sizes):
            zero_paddings[idx] = (kernel_size - 1) / 2

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv_layers = nn.ModuleList()
        # this conversion needs to happen because of some internal torch problem with numpy.int.32
        filter_amounts = [channels_in] + list(map(int, filter_amounts))

        for k in range(self.nr_conv_layers):
            self.conv_layers.extend([nn.Conv2d(in_channels=filter_amounts[k],
                                               out_channels=filter_amounts[k + 1],
                                               kernel_size=kernel_sizes[k],
                                               padding=zero_paddings[k]),
                                     nn.ReLU()])
            if self.maxpool_layer_pos[k] == 1:
                self.conv_layers.extend([nn.MaxPool2d(kernel_size=3, stride=2)])
            if self.dropout_prob_per_layer[k] > 0:
                self.conv_layers.extend([nn.Dropout2d(p=self.dropout_prob_per_layer[k])])

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.conv_layers:
            x = layer(x)
        return x


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
        self.activation = activation_function

    def forward(self, x):
        for lin_layer in self.linear_layers[:-1]:
            x = self.activation(lin_layer(x))
        x = self.linear_layers[-1](x)
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
        self.fc_net = NNLinear(d_in, d_out, nr_lin_layers,
                                            start_layer_exponent, activation_function)

        out_features = self.fc_net.linear_layers[-1].in_features
        self.fc_net.linear_layers = self.fc_net.linear_layers[:-1]
        self.action_value_layer = nn.Linear(in_features=out_features, out_features=self.fc_net.D_out)
        self.board_value_layer = nn.Linear(in_features=out_features,
                                           out_features=1)
        self.game_dim = game_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def extract_features(self, x):
        x = x.to(self.device)
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
        x.to(self.device)
        x = self.conv_net(x)
        x = x.view(-1, self.d_in)
        x = self.fc_net(x)

        pi = self.action_value_layer(x)  # batch_size x action_size
        v = self.board_value_layer(x)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


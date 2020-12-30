import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt

from collections import namedtuple
import random
from sklearn.manifold import TSNE
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import copy
import random
from inspect import signature


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def slice_kwargs(func, kwargs):
    sliced_kwargs = dict()
    for p in signature(func).parameters.values():
        if p in kwargs:
            sliced_kwargs[p.name] = kwargs.pop(p.name)
    return sliced_kwargs


def plot_scores(episode_scores, n_smooth):
    """
    Plots the scores (accumulated reward over one episode) for a RL agent
    :param episode_scores: list of the episode scores
    :param n_smooth: averaged over this number of episodes
    :return: plot
    """
    plt.figure(2)
    plt.clf()
    scores_t = torch.FloatTensor(episode_scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    average = [0]
    if len(scores_t) >= n_smooth:
        means = scores_t.unfold(0, n_smooth, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(n_smooth - 1), means))
        average = means.numpy()
        plt.plot(average)
    plt.title(
        "Average Score over last {} Episodes: {}".format(
            n_smooth, int(average[-1] * 10) / 10
        )
    )
    # plt.pause(0.001)  # pause a bit so that plots are updated


def plot_stats(curr_average, episode_won, n_smooth, plot_freq):
    """
    Plots the winning/losing ratio
    :param episode_scores:
    :param n_smooth:
    :return:
    """
    plt.figure(2)
    plt.clf()
    scores = np.array(episode_won)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    if len(scores) >= n_smooth:
        mean = np.mean(scores[-n_smooth:])
        curr_average.append(mean)
        x = [
            n_smooth + plot_freq * i for i in range(len(curr_average))
        ]  # x axis values
        xi = [i for i in range(0, len(x))]  # ticks for each x point
        selection_val = int(np.floor(len(xi) / 5)) + 1
        xi = [tick for tick in xi if tick % selection_val == 0]
        x = [tick for idx, tick in enumerate(x) if idx % selection_val == 0]
        plt.xticks(xi, x)
        plt.plot(curr_average)
        plt.title(
            "Average Win Percentage over last {} Episodes: {}".format(
                n_smooth, int(curr_average[-1] * 100) / 100
            )
        )
        plt.pause(0.003)  # pause a bit so that plots are updated
    return curr_average


def plot_stats_all(episode_won, end_episode):
    """
    Plots the winning/losing ratio
    :param episode_scores:
    :param n_smooth:
    :return:
    """
    plt.figure(3)
    plt.clf()
    scores_t = torch.FloatTensor(episode_won)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    average = [0]
    for i in range(1, end_episode + 1):
        means = scores_t.unfold(0, i, 1).mean(1).view(-1)
        average.append(list(means.numpy())[0])
    plt.plot(average)
    plt.title(
        "Average Win Percentage over last {} Episodes: {}".format(
            end_episode, int(average[-1] * 100) / 100
        )
    )
    plt.pause(0.001)  # pause a bit so that plots are updated


class ReplayMemory(object):
    """
    Stores a state-transition (s, a, s', r) quadruple
    for approximating Q-values with q(s, a) <- r + gamma * max_a' q(s', a') updates
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def visualize_features(n_points, environment, env_name):
    """
    Visualize with t-SNE the models representation of a board state
    :param n_points: number of points in t-SNE plot
    :param env_name: environment name for plotting
    :return: plot of t-SNE and plot of some board states
    """
    boards = []
    states = []
    model = environment.agents[0].model
    interrupt = False

    print("Acquiring features")
    for i in range(n_points):
        environment.reset()
        done = False

        while not done:
            _, done, won = environment.step()

            board = copy.deepcopy(environment.board)
            state = environment.agents[0].state_to_tensor()
            boards.append(board)
            states.append(state)
            if (
                environment.steps > 20
            ):  # break loops to obtain more diverse feature space
                break
            if len(states) >= n_points:  # interrupt simulation if enough states
                interrupt = True
                break
        if interrupt:
            break

    states = Variable(torch.cat(states))
    features = model.extract_features(states)
    action_values = F.sigmoid(model.lin_final(features))
    features = features.data.numpy()
    state_values = action_values.data.numpy().max(1)

    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    X_tsne = tsne.fit_transform(features)

    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)  # scale to interval (0, 1)

    # print out 10 randomly chosen board states with positions and state-values
    # and mark choice in embedding
    choice = np.random.choice(len(boards), 10, replace=False)
    for i, c in enumerate(choice):
        print(i, c)
        print(X_tsne[c])
        print(state_values[c])
        print_board(boards[c], same_figure=False)
        # plt.title("state value: {}, spatial: {}".format(state_values[c], X_tsne[c]))
        plt.savefig("{}{}.png".format(env_name, i))

    def plot_embedding(features, values, choice):
        """
        Plot an embedding
        :param features: vectors to be plotted
        :param values: value between 0 and 1 for respective color (e.g. state-value)
        :return: plot
        """
        fig, ax = plt.subplots()
        mymap = plt.cm.get_cmap("Spectral")
        sm = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []  # fake array for scalar mappable urrgh..
        for i in range(features.shape[0]):
            plt.plot(features[i, 0], features[i, 1], ".", color=mymap(values[i]))
        cb = plt.colorbar(sm)
        cb.set_label("Q-Values")
        for i in range(features.shape[0]):  # overwrite old plotted points
            if i in choice:
                plt.plot(features[i, 0], features[i, 1], "o", color="k")
        plt.xticks([]), plt.yticks([])
        plt.title("t-SNE of Board Evaluator Features")
        plt.show(block=False)

    print("Plotting t-SNE embedding")
    plot_embedding(X_tsne, state_values, choice)
    plt.savefig("{}-tsne.png".format(env_name))


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

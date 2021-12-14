from typing import Union, Optional, Any, List, Iterable

import numpy as np
from matplotlib import pyplot as plt

import torch
from inspect import signature
from dataclasses import dataclass

RNG = Union[np.random.RandomState, np.random.Generator]


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
            sliced_kwargs[p.name] = kwargs.pop_last(p.name)
    return sliced_kwargs


def rng_from_seed(
    seed: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None
):
    if not isinstance(seed, np.random.Generator):
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    return rng


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


@dataclass
class RollingMeter(object):
    """
    Computes and stores the average and current value
    """

    val: Union[int, float] = 0
    avg: Union[int, float] = 0
    sum: Union[int, float] = 0
    max: Union[int, float] = 0
    min: Union[int, float] = 0
    count: Union[int, float] = 0

    def push(self, val, n=1):
        self.val = val
        self.avg = self.sum / self.count
        self.sum += val * n
        self.max = max(self.max, val)
        self.min = min(self.min, val)
        self.count += n


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.leaf_pos = 0
        # build a flattened binary tree
        self.leaf_level = int(np.ceil(np.log2(capacity)) + 1)
        # tree holds the priority values
        self.prioritree = np.zeros(2 ** self.leaf_level - 1)
        # values holds the actual data entries
        self.values = np.full(capacity, np.nan, dtype=object)

    def __len__(self):
        return self.size

    def sum(self):
        return self.prioritree[0]

    def insert(self, value, priority: float):
        # write the new value to the data list and update the priority of the tree path
        self.values[self.leaf_pos] = value
        self.update(2 ** (self.leaf_level - 1) - 1 + self.leaf_pos, priority)

        self.leaf_pos += 1
        self.leaf_pos %= self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        delta = priority - self.prioritree[idx]
        self.prioritree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.prioritree[idx] += delta

    def priority(self, index: int):
        """
        Returns the priority of the value at the given index.

        Parameters
        ----------
        index: int,
            the index of the value

        Returns
        -------
        float,
            the priority
        """
        tree_index = 2 ** (self.leaf_level - 1) - 1 + index
        return self.prioritree[tree_index]

    def get(self, priority: float, percentage: bool = True):
        """
        Return the value corresponding to the given priority.

        Traverses the tree from top to bottom checking if the priority is found in the left subtree or in the right
        subtree. The recursion breaks once the a leaf is found
        """
        if percentage:
            priority *= self.prioritree[0]
        idx = 0
        breaking_idx = 2 ** (self.leaf_level - 1) - 1
        while True:
            left_idx = 2 * idx + 1
            if priority <= self.prioritree[left_idx]:
                idx = left_idx
            else:
                idx = left_idx + 1  # the right child's idx
                priority -= self.prioritree[left_idx]

            if (value_idx := idx - breaking_idx) >= 0:
                return value_idx, self.values[value_idx], self.prioritree[idx]

    def as_str(self):
        out = [[]]
        level = 1
        curr_elems = 2 ** level - 1
        for i in range(self.prioritree.shape[0]):
            out[level - 1].append(str(self.prioritree[i]))
            if i + 1 == curr_elems:
                out.append([])
                level += 1
                curr_elems = 2 ** level - 1
        return "\n".join([" ".join(stage) for stage in out])


class PERContainer(object):
    """
    A PRIORITIZED EXPERIENCE REPLAY container.
    Allows for sampling according to the probability distribution defined by the contained elements' priorities.

    References
     ---------
     .. [1] Tom Schaul, John Quan, Ioannis Antonoglou and David Silver,
        https://arxiv.org/pdf/1511.05952.pdf .
    """

    def __init__(
        self,
        capacity: int,
        alpha: float,
        seed: Union[np.random.Generator, np.random.RandomState, int, None] = None,
    ):
        r"""
        Initializes the sum tree of length ``capacity``.

        Parameters
        ----------
        capacity : int
            the maximum number of samples to memorize

        alpha: float
            the exponent to determine the degree of prioritization:

            .. math:: P_i \sim \mathrm{prio}_i ^ \alpha / \sum_i \mathrm{prio}_i ^ \alpha

            A lower :math:`\alpha` creates more uniformly distributed probability values.

        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self._alpha = alpha
        self.rng = rng_from_seed(seed)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        """
        Set the alpha parameter to a new value.
        """
        old_alpha = self._alpha
        self._alpha = alpha
        priorities = (
            self.tree.priority(i) ** -old_alpha for i in range(len(self.tree))
        )
        self.update(range(len(self.tree)), priorities)

    def push(self, value: Any):
        """
        Adds a new sample.

        Parameters
        ----------
        value: Any,
            the new sample value
        """
        self.tree.insert(value, float("inf"))

    def sample(self, n: int, beta: float):
        """
        Samples `n` times.

        Parameters
        ----------
        n: int,
            the number of samples
        beta: float,
            the temperature parameter of the distribution

        Returns
        -------
        out: Tuple containing
            [
                list of samples,
                list of weights,
                list of sample indices in the sum tree
            ]
        """
        out = []
        indices = []
        weights = []
        priorities = []
        n_samples = min(n, len(self.tree))
        rs = self.rng.uniform(size=n_samples)
        for i in range(n_samples):
            r = rs[i]
            data, priority, index = self.tree.get(r)
            priorities.append(priority)
            weights.append(
                (1.0 / self.capacity / priority) ** beta if priority > 1e-16 else 0
            )
            indices.append(index)
            out.append(data)
            self.update([index], [0])  # To avoid duplicating

        self.update(indices, priorities)  # Revert priorities

        weights /= max(weights)  # Normalize for stability

        return out, weights, indices

    def update(self, indices: Iterable[int], priorities: Iterable[float]):
        """
        Update each sample's priority (e.g. after an alpha change).

        Parameters
        ----------
        indices : Iterable[int],
            Iterable of sample indices to update
        priorities: Iterable[float],
            the new priorities for the given indices
        """
        for i, p in zip(indices, priorities):
            self.tree.update(i, p ** self._alpha)

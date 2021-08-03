from typing import Union, Optional, Any, List, Iterable

import numpy as np
from inspect import signature
from dataclasses import dataclass


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

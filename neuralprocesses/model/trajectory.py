import abc
import logging
from typing import Tuple

import lab as B
import numpy as np
import numpy.ma as ma
import torch

from neuralprocesses.dist import UniformDiscrete

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def permute(xi, seed=None):
    state = B.global_random_state(B.dtype(xi))
    B.random.set_global_random_state(state)
    if seed is not None:
        np.random.seed(seed)
    xi = xi[:, :, np.random.permutation(xi.shape[-1])]
    return xi


class AbstractTrajectoryGenerator(metaclass=abc.ABCMeta):
    """Abstract trajectory generator

    Attributes:
        trajectory_length (int): length of trajectory to generate
        min_x (float): minimum value of x
        max_x (float): maximum value of x
    """

    @abc.abstractmethod
    def generate(self, x_context):
        """Generate a trajectory

        Returns:

        """
        raise NotImplementedError("Abstract method")


class GridGenerator(AbstractTrajectoryGenerator):
    def __init__(self, trajectory_length: int, min_x: float, max_x: float):
        self.trajectory_length = trajectory_length
        self.min_x = min_x
        self.max_x = max_x

    def generate(self, x_context=None):
        xi = B.linspace(torch.float32, self.min_x, self.max_x, self.trajectory_length)[
            None, None, :
        ]
        xi = permute(xi)
        return xi


class RandomGenerator(AbstractTrajectoryGenerator):
    def __init__(self, trajectory_length: int, min_x: float, max_x: float):
        self.trajectory_length = trajectory_length
        self.min_x = min_x
        self.max_x = max_x

    def generate(self, x_context=None):
        xi = torch.distributions.Uniform(low=self.min_x, high=self.max_x).sample(
            [self.trajectory_length]
        )[None, None, :]
        xi = permute(xi)
        return xi


class EmanateGridGenerator(AbstractTrajectoryGenerator):
    """
    Create a grid, and then randomly pick one context point. Choose the order of
    x points on the grid by choosing the grid point closest to this chosen context.
    Then choose the following point which is closest the last chosen point. Continue
    until you've exhausted the grid.
    """

    def __init__(
        self,
        trajectory_length: int,
        min_x: float,
        max_x: float,
    ):
        self.trajectory_length = trajectory_length
        self.min_x = min_x
        self.max_x = max_x

    def generate(self, x_context, seed=None):
        xi = B.linspace(torch.float32, self.min_x, self.max_x, self.trajectory_length)[
            None, None, :
        ]
        xi = emanate(xi, x_context, seed=seed)
        return xi


def emanate(xi, x_context, seed=None):
    xin = xi[0, 0, :]
    state = np.random.RandomState(seed)
    state, coin = UniformDiscrete(1, x_context.shape[-1]).sample(state, np.int64)
    start = x_context[0, 0, coin - 1].item()

    i = 0
    next_x = start
    mask = np.zeros(xin.shape[0], dtype=bool)
    order = np.zeros(xin.shape[0], dtype=np.int64)
    while True:
        xim = ma.masked_array(xin, mask=mask)
        next_x_ind = ((xim - next_x) ** 2).argmin()
        next_x = xim[next_x_ind]
        mask[next_x_ind] = True
        order[i] = next_x_ind
        if sum(mask) == xin.shape[0]:
            break
        i += 1
    if len(np.unique(order)) != len(order):
        raise ValueError("Duplicate points in trajectory!")
    return xi[:, :, order]


def construct_trajectory_gens(
    trajectory_length: int, x_range: Tuple
):
    gens = {
        "grid": GridGenerator(trajectory_length, x_range[0], x_range[1]),
        "random": RandomGenerator(trajectory_length, x_range[0], x_range[1]),
        "emanate": EmanateGridGenerator(trajectory_length, x_range[0], x_range[1]),
    }
    return gens

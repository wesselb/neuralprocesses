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
    if seed is not None:
        torch.manual_seed(seed)
    # state = B.global_random_state(xi)
    perm_xi = torch.Tensor(xi.shape)
    # same values will be present in each row (corresponding to one function)
    # there will be a difference in the order for each one
    for ri in torch.arange(xi.shape[0]):
        perm_xi[ri, :, ] = xi[ri, :, torch.randperm(xi.shape[-1])]
    return perm_xi


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

    def generate(self, x_context=None, seed=None):
        grid = B.linspace(torch.float32, self.min_x, self.max_x, self.trajectory_length)
        rep_grid = grid.repeat((x_context.shape[0], 1)).reshape(x_context.shape[0], 1, self.trajectory_length)
        xi = permute(rep_grid, seed=seed)
        return xi


class RandomGenerator(AbstractTrajectoryGenerator):
    def __init__(self, trajectory_length: int, min_x: float, max_x: float):
        self.trajectory_length = trajectory_length
        self.min_x = min_x
        self.max_x = max_x

    def generate(self, x_context=None):
        xi = torch.distributions.Uniform(low=self.min_x, high=self.max_x).sample(
            [self.trajectory_length]
        )
        xi = xi.repeat((x_context.shape[0], 1)).reshape(x_context.shape[0], 1, self.trajectory_length)
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
        xi = B.linspace(self.min_x, self.max_x, self.trajectory_length)
        xi = torch.Tensor(xi).to(torch.float32).cpu()
        xi = xi.repeat((x_context.shape[0], 1)).reshape(x_context.shape[0], 1, self.trajectory_length)
        xi = emanate(xi, x_context, seed=seed)
        return xi


class EmanateRandomGenerator(AbstractTrajectoryGenerator):
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
        if seed is not None:
            torch.manual_seed(seed)
        xi = torch.distributions.Uniform(low=self.min_x, high=self.max_x).sample(
            [self.trajectory_length]
        )
        xi = xi.repeat((x_context.shape[0], 1)).reshape(x_context.shape[0], 1, self.trajectory_length)
        if x_context.nelement() == 0:
            LOG.warning("Cannot use emanate with no context, using random instead for this trajectory.")
            xi = permute(xi)
        else:
            xi = emanate(xi, x_context, seed=seed)
        return xi


def emanate(xi, x_context, seed=None):
    if seed is not None:
        seeds = [seed + s for s in np.arange(x_context.shape[0])]
    else:
        seeds = [None for _ in np.arange(x_context.shape[0])]
    for i, s in zip(np.arange(xi.shape[0]), seeds):
        xin = xi[i, 0, :]
        xin_context = x_context[i, 0, :]
        xin_ordered = inner_emanate(xin, xin_context, seed=s)
        xi[i, 0, :] = xin_ordered
    return xi


def inner_emanate(xin, xin_context, seed=None):
    state = np.random.RandomState(seed)
    state, coin = UniformDiscrete(1, xin_context.shape[0]).sample(state, np.int64)
    start = xin_context[coin - 1].item()

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
    return xin[order]


def construct_trajectory_gens(
    trajectory_length: int, x_range: Tuple
):
    gens = {
        "grid": GridGenerator(trajectory_length, x_range[0], x_range[1]),
        "random": RandomGenerator(trajectory_length, x_range[0], x_range[1]),
        "emanate": EmanateGridGenerator(trajectory_length, x_range[0], x_range[1]),
        "emanate-random": EmanateRandomGenerator(trajectory_length, x_range[0], x_range[1]),
    }
    return gens

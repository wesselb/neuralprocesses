import abc
import logging
from typing import Tuple

import lab as B
import numpy as np
import torch

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
    def generate(self):
        """Generate a trajectory

        Returns:

        """
        raise NotImplementedError("Abstract method")


class GridGenerator(AbstractTrajectoryGenerator):
    def __init__(self, trajectory_length: int, min_x: float, max_x: float):
        self.trajectory_length = trajectory_length
        self.min_x = min_x
        self.max_x = max_x

    def generate(self):
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

    def generate(self):
        xi = torch.distributions.Uniform(low=self.min_x, high=self.max_x).sample(
            [self.trajectory_length]
        )[None, None, :]
        xi = permute(xi)
        return xi


def construct_trajectory_gens(trajectory_length: int, x_range: Tuple):
    gens = {
        "grid": GridGenerator(trajectory_length, x_range[0], x_range[1]),
        "random": RandomGenerator(trajectory_length, x_range[0], x_range[1]),
    }
    return gens

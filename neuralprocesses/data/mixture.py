import lab as B
import numpy as np

from .data import AbstractGenerator, DataGenerator
from .. import _dispatch

__all__ = ["MixtureGenerator"]


class MixtureGenerator(AbstractGenerator):
    """A mixture of data generators.

    Args:
        *gens (:class:`.data.DataGenerator`): Components of the mixture.
        seed (int, optional): Random seed. Defaults to `0`.

    Attributes:
        gens (tuple[:class:`.data.SyntheticGenerator`]): Components of the mixture.
        state (random state): Random state.
    """

    @_dispatch
    def __init__(self, *gens: DataGenerator, seed=0):
        self.gens = gens
        self.state = B.create_random_state(np.float64, seed=seed)

    def generate_batch(self):
        self.state, i = B.randint(self.state, np.int64, upper=len(self.gens))
        return self.gens[i].generate_batch()

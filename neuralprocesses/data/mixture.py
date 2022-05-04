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
        num_batches (int): Number of batches in an epoch.
        gens (tuple[:class:`.data.SyntheticGenerator`]): Components of the mixture.
        state (random state): Random state.
    """

    @_dispatch
    def __init__(self, *gens: DataGenerator, seed=0):
        if not all(gens[0].num_batches == g.num_batches for g in gens[1:]):
            raise ValueError(
                f"Attribute `num_batches` inconsistent between elements of the mixture."
            )
        self.num_batches = gens[0].num_batches
        self.gens = gens
        self.state = B.create_random_state(np.float64, seed=seed)

    def generate_batch(self):
        self.state, i = B.randint(self.state, np.int64, upper=len(self.gens))
        return self.gens[i].generate_batch()

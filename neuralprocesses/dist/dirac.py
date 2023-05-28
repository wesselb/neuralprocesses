import lab as B
from plum import parametric
from wbml.util import indented_kv

from .. import _dispatch
from ..aggregate import Aggregate
from ..util import batch
from .dist import AbstractDistribution

__all__ = ["Dirac"]


@parametric
class Dirac(AbstractDistribution):
    """A Dirac delta.

    Args:
        x (tensor): Position of the Dirac delta.
        d (int): Dimensionality of the data.

    Attributes:
        x (tensor): Position of the Dirac delta.
        d (int): Dimensionality of the data.
    """

    def __init__(self, x, d):
        self.x = x
        self.d = d

    def __repr__(self):
        return "<Dirac:\n" + indented_kv("x", repr(self.x), suffix=">")

    def __str__(self):
        return "<Dirac:\n" + indented_kv("x", str(self.x), suffix=">")

    @property
    def mean(self):
        return self.x

    @property
    def var(self):
        return self._var()

    @_dispatch
    def _var(self: "Dirac[B.Numeric, B.Numeric]"):
        with B.on_device(self.x):
            return B.zeros(self.x)

    @_dispatch
    def _var(self: "Dirac[Aggregate, Aggregate]"):
        return Aggregate(*(Dirac(xi, di).var for xi, di in zip(self.x, self.d)))

    @_dispatch
    def logpdf(self: "Dirac[B.Numeric, B.Int]", x):
        with B.on_device(self.x):
            return B.zeros(B.dtype(self.x), *batch(self.x, self.d + 1))

    @_dispatch
    def logpdf(self: "Dirac[Aggregate, Aggregate]", x):
        # Just take the first one. It doesn't matter.
        return Dirac(self.x[0], self.d[0]).logpdf(None)

    @_dispatch
    def sample(
        self: "Dirac[B.Numeric, B.Int]",
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        # If no number of samples was specified, don't add a sample dimension.
        if shape == ():
            return state, self.x
        else:
            # Don't tile. This way is more efficient.
            return state, B.expand_dims(self.x, axis=0, times=len(shape))

    @_dispatch
    def sample(
        self: "Dirac[Aggregate, Aggregate]",
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        samples = []
        for xi, di in zip(self.x, self.d):
            state, sample = Dirac(xi, di).sample(state, dtype, *shape)
            samples.append(sample)
        return state, Aggregate(*samples)

    @_dispatch
    def kl(self, other: "Dirac"):
        # Same result as `logpdf`, so just reuse that method.
        return self.logpdf(None)


@B.dtype.dispatch
def dtype(dist: Dirac):
    return B.dtype(dist.x)

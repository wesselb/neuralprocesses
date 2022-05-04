import lab as B
from wbml.util import indented_kv

from .dist import AbstractMultiOutputDistribution
from .. import _dispatch
from ..util import batch
from ..aggregate import Aggregate

__all__ = ["Dirac"]


class Dirac(AbstractMultiOutputDistribution):
    """A Dirac delta.

    Also accepts aggregated of its arguments.

    Args:
        x (tensor): Position of the Dirac delta of shape `(*b, c, *n)`.
        d (int): Dimensionality of the data, i.e. `len(n)`.

    Attributes:
        x (tensor): Position of the Dirac delta of shape `(*b, c, *n)`.
        d (int): Dimensionality of the data, i.e. `len(n)`.
    """

    def __init__(self, x, d):
        self.x = x
        self.d = d

    def __repr__(self):
        return f"<Dirac:\n" + indented_kv("x", repr(self.x), suffix=">")

    def __str__(self):
        return f"<Dirac:\n" + indented_kv("x", str(self.x), suffix=">")

    @property
    def mean(self):
        return self.x

    @property
    def var(self):
        return self._var(self.x)

    @staticmethod
    @_dispatch
    def _var(x: B.Numeric):
        with B.on_device(x):
            return B.zeros(x)

    @staticmethod
    @_dispatch
    def _var(x: Aggregate):
        return Aggregate(*(Dirac._var(xi) for xi in x))

    def logpdf(self, x):
        return self._logpdf(self.x, self.d)

    @staticmethod
    @_dispatch
    def _logpdf(x: B.Numeric, d: B.Int):
        with B.on_device(x):
            return B.zeros(B.dtype(x), *batch(x, d + 1))

    @staticmethod
    @_dispatch
    def _logpdf(x: Aggregate, d: Aggregate):
        # Just take the first one. It doesn't matter.
        return Dirac._logpdf(x[0], d[0])

    @_dispatch
    def sample(self, num=1):
        return self._sample(self.x, num=num)

    @_dispatch
    def sample(self, state: B.RandomState, num=1):
        return state, self.sample(num=num)

    @staticmethod
    @_dispatch
    def _sample(x: B.Numeric, *, num):
        # If there is only one sample, squeeze the sample dimension, which happens
        # here by not adding it.
        if num == 1:
            return x
        else:
            # Don't tile. This way is more efficient.
            return x[None, ...]

    @staticmethod
    @_dispatch
    def _sample(x: Aggregate, *, num):
        return Aggregate(*(Dirac._sample(xi, num=num) for xi in x))

    @_dispatch
    def kl(self, other: "Dirac"):
        # Same result as `logpdf`, so just reuse that method.
        return self.logpdf(None)

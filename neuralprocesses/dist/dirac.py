import lab as B
from wbml.util import indented_kv

from .dist import AbstractDistribution
from .. import _dispatch
from ..aggregate import Aggregate
from ..util import batch

__all__ = ["Dirac"]


class Dirac(AbstractDistribution):
    """A Dirac delta.

    Also accepts aggregated of its arguments.

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
    def sample(self, num=None):
        return self._sample(self.x, num=num)

    @_dispatch
    def sample(self, state: B.RandomState, num=None):
        return state, self.sample(num=num)

    @staticmethod
    @_dispatch
    def _sample(x: B.Numeric, *, num):
        # If no number of samples was specified, don't add a sample dimension.
        if num is None:
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

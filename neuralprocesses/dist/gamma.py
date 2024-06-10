import lab as B
from matrix.shape import broadcast
from plum import parametric

from .. import _dispatch
from ..aggregate import Aggregate
from ..mask import Masked
from .dist import AbstractDistribution, shape_batch

__all__ = ["Gamma"]


@parametric
class Gamma(AbstractDistribution):
    """Gamma distribution.

    Args:
        k (tensor): Shape parameter.
        scale (tensor): Scale parameter.
        d (int): Dimensionality of the data.

    Attributes:
        k (tensor): Shape parameter.
        scale (tensor): Scale parameter.
        d (int): Dimensionality of the data.
    """

    def __init__(self, k, scale, d):
        self.k = k
        self.scale = scale
        self.d = d

    @property
    def mean(self):
        return B.multiply(self.k, self.scale)

    @property
    def var(self):
        return B.multiply(B.multiply(self.k, self.scale), self.scale)

    @_dispatch
    def sample(
        self: "Gamma[Aggregate, Aggregate, Aggregate]",
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        samples = []
        for ki, si, di in zip(self.k, self.scale, self.d):
            state, sample = Gamma(ki, si, di).sample(state, dtype, *shape)
            samples.append(sample)
        return state, Aggregate(*samples)

    @_dispatch
    def sample(
        self: "Gamma[B.Numeric, B.Numeric, B.Int]",
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        return B.randgamma(state, dtype, *shape, alpha=self.k, scale=self.scale)

    @_dispatch
    def logpdf(self: "Gamma[Aggregate, Aggregate, Aggregate]", x: Aggregate):
        return sum(
            [
                Gamma(ki, si, di).logpdf(xi)
                for ki, si, di, xi in zip(self.k, self.scale, self.d, x)
            ],
            0,
        )

    @_dispatch
    def logpdf(self: "Gamma[B.Numeric, B.Numeric, B.Int]", x: Masked):
        x, mask = x.y, x.mask
        with B.on_device(self.k):
            safe = B.to_active_device(B.one(B.dtype(self)))
        # Make inputs safe.
        x = mask * x + (1 - mask) * safe
        # Run with safe inputs, and filter out the right logpdfs.
        return self.logpdf(x, mask=mask)

    @_dispatch
    def logpdf(self: "Gamma[B.Numeric, B.Numeric, B.Int]", x: B.Numeric, *, mask=1):
        logz = B.loggamma(self.k) + self.k * B.log(self.scale)
        logpdf = (self.k - 1) * B.log(x) - x / self.scale - logz
        logpdf = logpdf * mask
        if self.d == 0:
            return logpdf
        else:
            return B.sum(logpdf, axis=tuple(range(B.rank(logpdf)))[-self.d :])

    def __str__(self):
        return f"Gamma({self.k}, {self.scale})"

    def __repr__(self):
        return f"Gamma({self.k!r}, {self.scale!r})"


@B.dtype.dispatch
def dtype(dist: Gamma):
    return B.dtype(dist.k, dist.scale)


@shape_batch.dispatch
def shape_batch(dist: "Gamma[B.Numeric, B.Numeric, B.Int]"):
    return B.shape_broadcast(dist.k, dist.scale)[: -dist.d]


@shape_batch.dispatch
def shape_batch(dist: "Gamma[Aggregate, Aggregate, Aggregate]"):
    return broadcast(
        *(
            shape_batch(Gamma(ki, si, di))
            for ki, si, di in zip(dist.k, dist.scale, dist.d)
        )
    )

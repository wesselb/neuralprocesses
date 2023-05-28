import lab as B
from matrix.shape import broadcast
from plum import parametric

from .. import _dispatch
from ..aggregate import Aggregate
from ..mask import Masked
from .dist import AbstractDistribution, shape_batch

__all__ = ["Beta"]


@parametric
class Beta(AbstractDistribution):
    """Beta distribution.

    Args:
        alpha (tensor): Shape parameter `alpha`.
        beta (tensor): Shape parameter `beta`.
        d (int): Dimensionality of the data.

    Attributes:
        alpha (tensor): Shape parameter `alpha`.
        beta (tensor): Shape parameter `beta`.
        d (int): Dimensionality of the data.
    """

    def __init__(self, alpha, beta, d):
        self.alpha = alpha
        self.beta = beta
        self.d = d

    @property
    def mean(self):
        return B.divide(self.alpha, B.add(self.alpha, self.beta))

    @property
    def var(self):
        sum = B.add(self.alpha, self.beta)
        with B.on_device(sum):
            one = B.one(sum)
        return B.divide(
            B.multiply(self.alpha, self.beta),
            B.multiply(B.multiply(sum, sum), B.add(sum, one)),
        )

    @_dispatch
    def sample(
        self: "Beta[Aggregate, Aggregate, Aggregate]",
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        samples = []
        for ai, bi, di in zip(self.alpha, self.beta, self.d):
            state, sample = Beta(ai, bi, di).sample(state, dtype, *shape)
            samples.append(sample)
        return state, Aggregate(*samples)

    @_dispatch
    def sample(
        self: "Beta[B.Numeric, B.Numeric, B.Int]",
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        return B.randbeta(state, dtype, *shape, alpha=self.alpha, beta=self.beta)

    @_dispatch
    def logpdf(self: "Beta[Aggregate, Aggregate, Aggregate]", x: Aggregate):
        return sum(
            [
                Beta(ai, bi, di).logpdf(xi)
                for ai, bi, di, xi in zip(self.alpha, self.beta, self.d, x)
            ],
            0,
        )

    @_dispatch
    def logpdf(self: "Beta[B.Numeric, B.Numeric, B.Int]", x: Masked):
        x, mask = x.y, x.mask
        with B.on_device(self.alpha):
            safe = B.to_active_device(B.cast(B.dtype(self.alpha), 0.5))
        # Make inputs safe.
        x = mask * x + (1 - mask) * safe
        # Run with safe inputs, and filter out the right logpdfs.
        return self.logpdf(x, mask=mask)

    @_dispatch
    def logpdf(self: "Beta[B.Numeric, B.Numeric, B.Int]", x: B.Numeric, *, mask=1):
        logz = B.logbeta(self.alpha, self.beta)
        logpdf = (self.alpha - 1) * B.log(x) + (self.beta - 1) * B.log(1 - x) - logz
        return B.sum(mask * logpdf, axis=tuple(range(B.rank(logpdf)))[-self.d :])

    def __str__(self):
        return f"Beta({self.alpha}, {self.beta})"

    def __repr__(self):
        return f"Beta({self.alpha!r}, {self.beta!r})"


@B.dtype.dispatch
def dtype(dist: Beta):
    return B.dtype(dist.alpha, dist.beta)


@shape_batch.dispatch
def shape_batch(dist: "Beta[B.Numeric, B.Numeric, B.Int]"):
    return B.shape_broadcast(dist.alpha, dist.beta)[: -dist.d]


@shape_batch.dispatch
def shape_batch(dist: "Beta[Aggregate, Aggregate, Aggregate]"):
    return broadcast(
        *(
            shape_batch(Beta(ai, bi, di))
            for ai, bi, di in zip(dist.alpha, dist.beta, dist.d)
        )
    )

import lab as B
from matrix.shape import broadcast

from .dist import AbstractDistribution, shape_batch
from .. import _dispatch
from ..aggregate import Aggregate
from ..mask import Masked

__all__ = ["Beta"]


class Beta(AbstractDistribution):
    """Beta distribution.

    Args:
        alpha (tensor): Shape parameter `alpha`.
        beta (tensor): Shape parameter `beta`.
        d (int): Dimensionality of the sample.

    Attributes:
        alpha (tensor): Shape parameter `alpha`.
        beta (tensor): Shape parameter `beta`.
        d (int): Dimensionality of the sample.
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
        self,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        return self.sample(self.alpha, self.beta, state, dtype, *shape)

    @_dispatch
    def sample(
        self,
        alpha: Aggregate,
        beta: Aggregate,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        samples = []
        for ai, bi in zip(alpha, beta):
            state, sample = self.sample(ai, bi, state, dtype, *shape)
            samples.append(sample)
        return state, Aggregate(*samples)

    @_dispatch
    def sample(
        self,
        alpha: B.Numeric,
        beta: B.Numeric,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        return B.randbeta(state, dtype, *shape, alpha=alpha, beta=beta)

    @_dispatch
    def logpdf(self, x):
        return self.logpdf(self.alpha, self.beta, self.d, x)

    @_dispatch
    def logpdf(self, alpha: Aggregate, beta: Aggregate, d: Aggregate, x: Aggregate):
        return sum(
            [self.logpdf(ai, bi, di, xi) for ai, bi, di, xi in zip(alpha, beta, d, x)],
            0,
        )

    @_dispatch
    def logpdf(self, alpha: B.Numeric, beta: B.Numeric, d: B.Int, x: Masked):
        x, mask = x.y, x.mask
        with B.on_device(alpha):
            safe = B.to_active_device(B.cast(B.dtype(alpha), 0.5))
        # Make inputs safe.
        x = mask * x + (1 - mask) * safe
        # Run with safe inputs, and filter out the right logpdfs.
        return self.logpdf(alpha, beta, d, x, mask=mask)

    @_dispatch
    def logpdf(
        self,
        alpha: B.Numeric,
        beta: B.Numeric,
        d: B.Int,
        x: B.Numeric,
        *,
        mask=1,
    ):
        logz = B.logbeta(alpha, beta)
        logpdf = (alpha - 1) * B.log(x) + (beta - 1) * B.log(1 - x) - logz
        return B.sum(mask * logpdf, axis=tuple(range(B.rank(logpdf)))[-d:])

    def __str__(self):
        return f"Beta({self.alpha}, {self.beta})"

    def __repr__(self):
        return f"Beta({self.alpha!r}, {self.beta!r})"


@B.dtype.dispatch
def dtype(dist: Beta):
    return B.dtype(dist.alpha, dist.beta)


@shape_batch.dispatch
def shape_batch(dist: Beta):
    return shape_batch(dist, dist.alpha, dist.beta, dist.d)


@shape_batch.dispatch
def shape_batch(dist: Beta, alpha: B.Numeric, beta: B.Numeric, d: B.Int):
    return B.shape_broadcast(alpha, beta)[:-d]


@shape_batch.dispatch
def shape_batch(dist: Beta, alpha: Aggregate, beta: Aggregate, d: Aggregate):
    return broadcast(
        *(shape_batch(dist, ai, bi, di) for ai, bi, di in zip(alpha, beta, d))
    )

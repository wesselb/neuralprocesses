import lab as B

from .dist import AbstractDistribution
from .. import _dispatch
from ..aggregate import Aggregate

__all__ = ["Beta"]


class Beta(AbstractDistribution):
    """Beta distribution.

    Args:
        alpha (tensor): Shape parameter `alpha`.
        beta (tensor): Shape parameter `beta`.

    Attributes:
        alpha (tensor): Shape parameter `alpha`.
        beta (tensor): Shape parameter `beta`.
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self):
        return B.divide(self.alpha, B.add(self.alpha, self.beta))

    @property
    def var(self):
        sum = B.add(self.alpha, self.beta)
        with B.on_device(sum):
            return B.divide(
                B.multiply(self.alpha, self.beta),
                B.multiply(B.multiply(sum, sum), B.add(sum, B.one(sum))),
            )

    @property
    def m1(self):
        return self.mean

    @property
    def m2(self):
        mean = self.mean
        return B.add(self.var, B.multiply(mean, mean))

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
        # TODO: This behaviour is wrong! Refactor!
        return self.logpdf(self.alpha, self.beta, x)

    @_dispatch
    def logpdf(self, alpha: Aggregate, beta: Aggregate, x: Aggregate):
        return Aggregate(
            *(self.logpdf(ai, bi, xi) for ai, bi, xi in zip(alpha, beta, x))
        )

    @_dispatch
    def logpdf(self, alpha: B.Numeric, beta: B.Numeric, x: B.Numeric):
        return (
            (alpha - 1) * B.log(x) + (beta - 1) * B.log(1 - x) - B.logbeta(alpha, beta)
        )

    def __str__(self):
        return f"Beta({self.alpha}, {self.beta})"

    def __repr__(self):
        return f"Beta({self.alpha!r}, {self.beta!r})"


@B.dtype.dispatch
def dtype(d: Beta):
    return B.dtype(d.alpha, d.beta)


@B.shape.dispatch
def shape(d: Beta):
    return B.shape_broadcast(d.alpha, d.beta)

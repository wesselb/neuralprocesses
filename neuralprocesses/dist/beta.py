import lab as B

from .dist import AbstractDistribution
from .. import _dispatch

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

    def __init__(self, alpha: B.Numeric, beta: B.Numeric):
        self.alpha = alpha
        self.beta = beta

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
        return B.randbeta(state, dtype, *shape, alpha=self.alpha, beta=self.beta)

    @_dispatch
    def logpdf(self, x):
        return (
            (self.alpha - 1) * B.log(x)
            + (self.beta - 1) * B.log(1 - x)
            - B.logbeta(self.alpha, self.beta)
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
    return B.shape(d.alpha, d.beta)

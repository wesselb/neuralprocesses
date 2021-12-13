import lab as B
import matrix  # noqa

from . import _dispatch
from .util import abstract

__all__ = ["InputsCoder", "AbstractDeepSet"]


class InputsCoder:
    """Encode with the target inputs."""


@_dispatch
def code(coder: InputsCoder, xz, z, x):
    return x, x


@abstract
class AbstractDeepSet:
    """Deep set.

    Args:
        phi (function): Pre-aggregation function.
        rho (function): Post-aggregation function.
        agg (function, optional): Aggregation function. Defaults to taking the mean.
    """

    def __init__(
        self,
        phi,
        rho,
        agg=lambda x: B.mean(x, axis=2, squeeze=False),
    ):
        self.phi = phi
        self.rho = rho
        self.agg = agg


@_dispatch
def code(coder: AbstractDeepSet, xz, z, x):
    z = B.concat(xz, z, axis=1)
    z = coder.phi(z)
    z = coder.agg(z)  # This aggregates over the data dimension.
    z = coder.rho(z)
    return x, z

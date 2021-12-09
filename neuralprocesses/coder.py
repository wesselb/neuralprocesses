import lab as B
import matrix  # noqa

from . import _dispatch
from .parallel import Parallel

__all__ = ["InputsCoder", "DeepSet"]


class InputsCoder:
    pass


@_dispatch
def code(coder: InputsCoder, xz, z, x):
    return x, x


class DeepSet:
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
def code(coder: DeepSet, xz, z, x):
    z = B.concat(xz, z, axis=1)
    z = coder.phi(z)
    z = coder.agg(z)  # This aggregates over the data dimension.
    z = coder.rho(z)
    return x, z

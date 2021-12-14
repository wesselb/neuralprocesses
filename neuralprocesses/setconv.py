import lab as B

from . import _dispatch
from .util import abstract

__all__ = ["AbstractSetConv1dEncoder", "AbstractSetConv1dDecoder"]


@abstract
class AbstractSetConv1dEncoder:
    def __init__(self, points_per_unit):
        self.log_scale = self.nn.Parameter(B.log(2 / points_per_unit))


@_dispatch
def code(encoder: AbstractSetConv1dEncoder, xz, z, x):
    # Prepend density channel.
    with B.on_device(z):
        density_channel = B.ones(B.dtype(z), B.shape(z, 0), 1, B.shape(z, 2))
    z = B.concat(density_channel, z, axis=1)

    # Compute interpolation weights.
    dists2 = B.pw_dists2(B.transpose(xz), B.transpose(x))
    weights = B.exp(-0.5 * dists2 / B.exp(2 * encoder.log_scale))

    # Interpolate to grid.
    z = B.matmul(z, weights)

    # Normalise by density channel.
    z = B.concat(z[:, :1, :], z[:, 1:, :] / (z[:, :1, :] + 1e-8), axis=1)

    return x, z


@abstract
class AbstractSetConv1dDecoder:
    def __init__(self, points_per_unit):
        self.log_scale = self.nn.Parameter(B.log(2 / points_per_unit))


@_dispatch
def code(encoder: AbstractSetConv1dDecoder, xz, z, x):
    # Compute interpolation weights.
    dists2 = B.pw_dists2(B.transpose(xz), B.transpose(x))
    weights = B.exp(-0.5 * dists2 / B.exp(2 * encoder.log_scale))

    # Interpolate to `x`.
    z = B.matmul(z, weights)

    return x, z

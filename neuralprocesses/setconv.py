from string import ascii_lowercase as letters

import lab as B

from . import _dispatch
from .util import abstract

__all__ = ["AbstractSetConv"]


@abstract
class AbstractSetConv:
    def __init__(self, points_per_unit, density_channel=False):
        self.log_scale = self.nn.Parameter(B.log(2 / points_per_unit))
        self.density_channel = density_channel


@_dispatch
def code(encoder: AbstractSetConv, xz, z, x):
    # Prepend density channel.
    if encoder.density_channel:
        with B.on_device(z):
            density_channel = B.ones(B.dtype(z), B.shape(z, 0), 1, *B.shape(z)[2:])
        z = B.concat(density_channel, z, axis=1)

    # Perform smoothing.
    z = smooth(encoder, xz, z, x)

    # Normalise by density channel.
    if encoder.density_channel:
        z = B.concat(z[:, :1, ...], z[:, 1:, ...] / (z[:, :1, ...] + 1e-8), axis=1)

    return x, z


def compute_weights(encoder, x1, x2):
    # Compute interpolation weights.
    dists2 = B.pw_dists2(B.transpose(x1), B.transpose(x2))
    return B.exp(-0.5 * dists2 / B.exp(2 * encoder.log_scale))


@_dispatch
def smooth(encoder: AbstractSetConv, xz: B.Numeric, z: B.Numeric, x: B.Numeric):
    return B.matmul(z, compute_weights(encoder, xz, x))


@_dispatch
def smooth(encoder: AbstractSetConv, xz: B.Numeric, z: B.Numeric, x: tuple):
    ws = [compute_weights(encoder, xz[:, i : i + 1, :], xi) for i, xi in enumerate(x)]
    letters_i = 3
    base = "abc"
    result = "ab"
    for _ in x:
        let = letters[letters_i]
        base += f",ac{let}"
        result += f"{let}"
        letters_i += 1
    return B.einsum(f"{base}->{result}", z, *ws)


@_dispatch
def smooth(encoder: AbstractSetConv, xz: tuple, z: B.Numeric, x: B.Numeric):
    ws = [compute_weights(encoder, xzi, x[:, i : i + 1, :]) for i, xzi in enumerate(xz)]
    letters_i = 3
    base_base = "ab"
    base_els = ""
    for _ in xz:
        let = letters[letters_i]
        base_base += f"{let}"
        base_els += f",a{let}c"
        letters_i += 1
    return B.einsum(f"{base_base}{base_els}->abc", z, *ws)

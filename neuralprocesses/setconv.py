from string import ascii_lowercase as letters

import lab as B

from . import _dispatch
from .util import register_module

__all__ = [
    "SetConv",
    "PrependDensityChannel",
    "DivideByFirstChannel",
    "AppendHarmonics",
]


@register_module
class SetConv:
    def __init__(self, points_per_unit, dtype=None):
        self.log_scale = self.nn.Parameter(B.log(2 / points_per_unit), dtype=dtype)


def compute_weights(encoder, x1, x2):
    # Compute interpolation weights.
    dists2 = B.pw_dists2(B.transpose(x1), B.transpose(x2))
    return B.exp(-0.5 * dists2 / B.exp(2 * encoder.log_scale))


@_dispatch
def code(encoder: SetConv, xz: B.Numeric, z: B.Numeric, x: B.Numeric):
    return x, B.matmul(z, compute_weights(encoder, xz, x))


@_dispatch
def code(encoder: SetConv, xz: B.Numeric, z: B.Numeric, x: tuple):
    ws = [compute_weights(encoder, xz[:, i : i + 1, :], xi) for i, xi in enumerate(x)]
    letters_i = 3
    base = "abc"
    result = "ab"
    for _ in x:
        let = letters[letters_i]
        base += f",ac{let}"
        result += f"{let}"
        letters_i += 1
    return x, B.einsum(f"{base}->{result}", z, *ws)


@_dispatch
def code(encoder: SetConv, xz: tuple, z: B.Numeric, x: B.Numeric):
    # Implement batching to allow decoding at very high resolution.
    batch_size = 1024
    if B.shape(x, 2) > batch_size:
        i = 0
        outs = []
        while i < B.shape(x, 2):
            outs.append(code(encoder, xz, z, x[:, :, i : i + batch_size])[1])
            i += batch_size
        return x, B.concat(*outs, axis=2)

    ws = [compute_weights(encoder, xzi, x[:, i : i + 1, :]) for i, xzi in enumerate(xz)]
    letters_i = 3
    base_base = "ab"
    base_els = ""
    for _ in xz:
        let = letters[letters_i]
        base_base += f"{let}"
        base_els += f",a{let}c"
        letters_i += 1
    return x, B.einsum(f"{base_base}{base_els}->abc", z, *ws)


@register_module
class PrependDensityChannel:
    def __call__(self, z):
        with B.on_device(z):
            density_channel = B.ones(B.dtype(z), B.shape(z, 0), 1, *B.shape(z)[2:])
        return B.concat(density_channel, z, axis=1)


@register_module
class DivideByFirstChannel:
    def __call__(self, z):
        return B.concat(z[:, :1, ...], z[:, 1:, ...] / (z[:, :1, ...] + 1e-8), axis=1)


@register_module
class AppendHarmonics:
    def __init__(self, x_range, num_harmonics):
        self.x_range = x_range
        self.num_harmonics = num_harmonics


@_dispatch
def code(encoder: AppendHarmonics, xz: B.Numeric, z: B.Numeric, x: B.Numeric):
    if B.shape(xz, 1) != 1:
        raise NotImplementedError(
            "`AppendHarmonics` currently only works for one-dimensional inputs."
        )
    lower, upper = encoder.x_range
    xz_on_interval = B.pi * (xz - lower) / (upper - lower)
    sines = B.concat(
        *[B.sin(i * xz_on_interval) for i in range(encoder.num_harmonics)], axis=1
    )
    cosines = B.concat(
        *[B.cos(i * xz_on_interval) for i in range(encoder.num_harmonics)], axis=1
    )
    return xz, B.concat(z, sines, cosines, axis=1)

from functools import wraps
from string import ascii_lowercase as letters
from typing import Optional

import lab as B

from . import _dispatch
from .augment import AugmentedInput
from .mask import Masked
from .parallel import Parallel
from .util import register_module

__all__ = [
    "SetConv",
    "PrependDensityChannel",
    "DivideByFirstChannel",
]


@register_module
class SetConv:
    """A set convolution.

    Args:
        scale (float): Initial value for the length scale.
        dtype (dtype, optional): Data type.

    Attributes:
        log_scale (scalar): Logarithm of the length scale.

    """

    def __init__(self, scale, dtype=None):
        self.log_scale = self.nn.Parameter(B.log(scale), dtype=dtype)


def _concrete_dim(x, i):
    try:
        int(B.shape(x, 2))
        return True
    except TypeError:
        return False


def _batch_targets(f):
    @wraps(f)
    def f_wrapped(coder, xz, z, x, batch_size=1024, **kw_args):
        if _concrete_dim(x, 2) and B.shape(x, 2) > batch_size:
            i = 0
            outs = []
            while i < B.shape(x, 2):
                outs.append(
                    code(
                        coder,
                        xz,
                        z,
                        x[:, :, i : i + batch_size],
                        batch_size=batch_size,
                        **kw_args,
                    )[1]
                )
                i += batch_size
            return x, B.concat(*outs, axis=2)
        else:
            return f(coder, xz, z, x, **kw_args)

    return f_wrapped


def compute_weights(coder, x1, x2):
    # Compute interpolation weights.
    dists2 = B.pw_dists2(B.transpose(x1), B.transpose(x2))
    return B.exp(-0.5 * dists2 / B.exp(2 * coder.log_scale))


@_dispatch
@_batch_targets
def code(coder: SetConv, xz: B.Numeric, z: B.Numeric, x: B.Numeric, **kw_args):
    return x, B.matmul(z, compute_weights(coder, xz, x))


@_dispatch
def code(coder: SetConv, xz: B.Numeric, z: B.Numeric, x: tuple, **kw_args):
    ws = [compute_weights(coder, xz[:, i : i + 1, :], xi) for i, xi in enumerate(x)]
    letters_i = 3
    base = "abc"
    result = "ab"
    for _ in x:
        let = letters[letters_i]
        letters_i += 1
        base += f",ac{let}"
        result += f"{let}"
    return x, B.einsum(f"{base}->{result}", z, *ws)


@_dispatch
@_batch_targets
def code(coder: SetConv, xz: tuple, z: B.Numeric, x: B.Numeric, **kw_args):
    ws = [compute_weights(coder, xzi, x[:, i : i + 1, :]) for i, xzi in enumerate(xz)]
    letters_i = 3
    base_base = "ab"
    base_els = ""
    for _ in xz:
        let = letters[letters_i]
        letters_i += 1
        base_base += f"{let}"
        base_els += f",a{let}c"
    return x, B.einsum(f"{base_base}{base_els}->abc", z, *ws)


@_dispatch
def code(coder: SetConv, xz: tuple, z: B.Numeric, x: tuple, **kw_args):
    ws = [compute_weights(coder, xzi, xi) for xzi, xi in zip(xz, x)]
    letters_i = 2
    base_base = "ab"
    base_els = ""
    result = "ab"
    for _ in x:
        let1 = letters[letters_i]
        letters_i += 1
        let2 = letters[letters_i]
        letters_i += 1
        base_base += f"{let1}"
        base_els += f",a{let1}{let2}"
        result += f"{let2}"
    return x, B.einsum(f"{base_base}{base_els}->{result}", z, *ws)


@_dispatch
def code(coder: SetConv, xz: Parallel, z: Parallel, x, **kw_args):
    xzs, zs = zip(*(code(coder, xzi, zi, x, **kw_args) for xzi, zi in zip(xz, z)))
    return Parallel(*xzs), Parallel(*zs)


@_dispatch
def code(coder: SetConv, xz, z, x: AugmentedInput, **kw_args):
    xz, z = code(coder, xz, z, x.x)
    return AugmentedInput(xz, x.augmentation), z


@register_module
class PrependDensityChannel:
    """Prepend a density channel to the current encoding."""

    @_dispatch
    def __call__(self, z: B.Numeric):
        with B.on_device(z):
            density_channel = B.ones(B.dtype(z), B.shape(z, 0), 1, *B.shape(z)[2:])
        return B.concat(density_channel, z, axis=1)

    @_dispatch
    def __call__(self, z: Parallel):
        return Parallel(*(self(zi) for zi in z))

    @_dispatch
    def __call__(self, z: Masked):
        # Apply mask to density channel _and_ the data channels. Since the mask has
        # only one channel, we can simply pointwise multiply and broadcasting should
        # do the rest for us.
        return z.mask * self(z.y)


@register_module
class DivideByFirstChannel:
    """Divide by the first channel.

    Args:
        epsilon (float): Value to add to the channel before dividing.

    Attributes:
        epsilon (float): Value to add to the channel before dividing.
    """

    @_dispatch
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon


@_dispatch
def code(
    coder: DivideByFirstChannel,
    xz,
    z: B.Numeric,
    x,
    epsilon: Optional[float] = None,
    **kw_args,
):
    epsilon = epsilon or coder.epsilon
    return (
        xz,
        B.concat(z[:, :1, ...], z[:, 1:, ...] / (z[:, :1, ...] + epsilon), axis=1),
    )


@_dispatch
def code(
    coder: DivideByFirstChannel,
    xz: Parallel,
    z: Parallel,
    x,
    **kw_args,
):
    xzs, zs = zip(*(code(coder, xzi, zi, x, **kw_args) for xzi, zi in zip(xz, z)))
    return Parallel(*xzs), Parallel(*zs)

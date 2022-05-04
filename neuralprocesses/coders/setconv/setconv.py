from functools import wraps
from string import ascii_lowercase as letters

import lab as B

from ... import _dispatch
from ...augment import AugmentedInput
from ...parallel import broadcast_coder_over_parallel
from ...util import register_module

__all__ = ["SetConv"]


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


def _dim_is_concrete(x, i):
    try:
        int(B.shape(x, i))
        return True
    except TypeError:
        return False


def _batch_targets(f):
    @wraps(f)
    def f_wrapped(coder, xz, z, x, batch_size=1024, **kw_args):
        # If `x` is the internal discretisation and we're compiling this function
        # with `tf.function`, then `B.shape(x, -1)` will be `None`. We therefore
        # check that `B.shape(x, -1)` is concrete before attempting the comparison.
        if _dim_is_concrete(x, -1) and B.shape(x, -1) > batch_size:
            i = 0
            outs = []
            while i < B.shape(x, -1):
                outs.append(
                    code(
                        coder,
                        xz,
                        z,
                        x[..., i : i + batch_size],
                        batch_size=batch_size,
                        **kw_args,
                    )[1]
                )
                i += batch_size
            return x, B.concat(*outs, axis=-1)
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


def _standardise_equation(equation):
    normalisation_map = {}
    for i, x in enumerate(equation):
        if x in letters and x not in normalisation_map:
            normalisation_map[x] = letters[len(normalisation_map)]
    normalised_equation = ""
    for x in equation:
        if x in letters:
            normalised_equation += normalisation_map[x]
        else:
            normalised_equation += x
    return normalised_equation


_setconv_cache_num_tup = {}


@_dispatch
def code(coder: SetConv, xz: B.Numeric, z: B.Numeric, x: tuple, **kw_args):
    ws = [compute_weights(coder, xz[..., i : i + 1, :], xi) for i, xi in enumerate(x)]

    # Use a cache so we don't build and normalise the string every time.
    try:
        equation = _setconv_cache_num_tup[len(x)]
    except KeyError:
        letters_i = 3
        base = "...bc"
        result = "...b"
        for _ in range(len(x)):
            let = letters[letters_i]
            letters_i += 1
            base += f",...c{let}"
            result += f"{let}"
        _setconv_cache_num_tup[len(x)] = _standardise_equation(f"{base}->{result}")

        equation = _setconv_cache_num_tup[len(x)]

    return x, _einsum(equation, z, *ws)


_setconv_cache_tup_num = {}


@_dispatch
@_batch_targets
def code(coder: SetConv, xz: tuple, z: B.Numeric, x: B.Numeric, **kw_args):
    ws = [compute_weights(coder, xzi, x[..., i : i + 1, :]) for i, xzi in enumerate(xz)]

    # Use a cache so we don't build and normalise the string every time.
    try:
        equation = _setconv_cache_tup_num[len(xz)]
    except KeyError:
        letters_i = 3
        base_base = "...b"
        base_els = ""
        for _ in range(len(xz)):
            let = letters[letters_i]
            letters_i += 1
            base_base += f"{let}"
            base_els += f",...{let}c"
        _setconv_cache_tup_num[len(xz)] = _standardise_equation(
            f"{base_base}{base_els}->...bc"
        )

        equation = _setconv_cache_tup_num[len(xz)]

    return x, _einsum(equation, z, *ws)


_setconv_cache_tup_tup = {}


@_dispatch
def code(coder: SetConv, xz: tuple, z: B.Numeric, x: tuple, **kw_args):
    ws = [compute_weights(coder, xzi, xi) for xzi, xi in zip(xz, x)]

    # Use a cache so we don't build and normalise the string every time.
    try:
        equation = _setconv_cache_tup_tup[len(x)]
    except KeyError:
        letters_i = 2
        base_base = "...b"
        base_els = ""
        result = "...b"
        for _ in range(len(x)):
            let1 = letters[letters_i]
            letters_i += 1
            let2 = letters[letters_i]
            letters_i += 1
            base_base += f"{let1}"
            base_els += f",...{let1}{let2}"
            result += f"{let2}"
        _setconv_cache_tup_tup[len(x)] = _standardise_equation(
            f"{base_base}{base_els}->{result}"
        )

        equation = _setconv_cache_tup_tup[len(x)]

    return x, _einsum(equation, z, *ws)


broadcast_coder_over_parallel(SetConv)


@_dispatch
def code(coder: SetConv, xz, z, x: AugmentedInput, **kw_args):
    xz, z = code(coder, xz, z, x.x, **kw_args)
    return AugmentedInput(xz, x.augmentation), z


def _einsum(equation, *args):
    """Even though `B.einsum` uses `opt_einsum`, further speed-ups are possible by
    appropriately using broadcasting or matrix multiplications. This function catches
    a few of those speed-ups.

    Args:
        equation (str): Equation.
        *args (tensor): Arguments.

    Returns:
        tensor: Result.
    """
    if equation == "...ab,...bc,...bd->...acd":
        x, y, z = args
        return B.sum(
            x[..., :, :, None, None]
            * y[..., None, :, :, None]
            * z[..., None, :, None, :],
            axis=-3,
        )
    elif equation == "...abc,...bd,...cd->...ad":
        x, y, z = args
        w = B.matmul(x, y[..., None, :, :], tr_a=True)
        return B.sum(w * z[..., None, :, :], axis=-2)
    elif equation == "...abc,...bd,...ce->...ade":
        x, y, z = args
        return B.matmul(y[..., None, :, :], x, z[..., None, :, :], tr_a=True)
    else:
        # Could not find a speed-up. Just use `B.einsum`, which uses `opt_einsum`
        # to find a good contraction order.
        return B.einsum(equation, *args)

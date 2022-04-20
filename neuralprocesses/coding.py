import lab as B
import matrix  # noqa

from . import _dispatch
from .parallel import Parallel
from .util import register_module, data_dims

__all__ = ["code", "Materialise"]


@_dispatch
def code(coder, xz, z, x, **kw_args):
    """Perform a coding operation.

    The default behaviour is to apply `coder` to `z` and return `(xz, coder(z))`.

    Args:
        coder (coder): Coder.
        xz (input): Current inputs corresponding to current encoding.
        z (tensor): Current encoding.
        x (input): Desired inputs.

    Returns:
        tuple[input, tensor]: New encoding.
    """
    return xz, coder(z)


@_dispatch
def _merge(z: B.Numeric):
    return z


@_dispatch
def _merge(zs: Parallel):
    return _merge(*zs)


@_dispatch
def _merge(z0: B.Numeric, *zs: B.Numeric):
    zs = (z0,) + zs
    # Remove all `None`s: those correspond to global features.
    zs = [z for z in zs if z is not None]
    if len(zs) == 0:
        raise ValueError("No inputs specified.")
    elif len(zs) > 1:
        diffs = sum([B.mean(B.abs(zs[0] - z)) for z in zs[1:]])
        if B.jit_to_numpy(diffs) > B.epsilon:
            raise ValueError("Cannot merge inputs.")
    return zs[0]


@_dispatch
def _merge(z0: tuple, *zs: tuple):
    zs = (z0,) + zs
    return tuple(_merge(*zis) for zis in zip(*zs))


@_dispatch
def _repeat_concat(xz: B.Numeric, z: B.Numeric):
    return z


@_dispatch
def _repeat_concat(xz: Parallel, z: Parallel):
    return _repeat_concat_parallel(*z, dims=data_dims(xz))


@_dispatch
def _repeat_concat_parallel(z0: B.Numeric, *zs: B.Numeric, dims):
    zs = (z0,) + zs
    # Broadcast the data dimensions. There are `dims` many of them, so perform a loop.
    shapes = [list(B.shape(z)) for z in zs]
    for i in range(B.rank(z0) - 1, B.rank(z0) - 1 - dims, -1):
        shape_n = max(shape[i] for shape in shapes)
        for shape in shapes:
            shape[i] = shape_n
    zs = [B.broadcast_to(z, *shape) for (z, shape) in zip(zs, shapes)]
    return B.concat(*zs, axis=-1 - dims)


@register_module
class Materialise:
    """Materialise an aggregate encoding."""


@_dispatch
def code(coder: Materialise, xz, z, x, **kw_args):
    return _merge(xz), _repeat_concat(xz, z)

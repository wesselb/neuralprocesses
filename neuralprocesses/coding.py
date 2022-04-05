import lab as B
import matrix  # noqa

from . import _dispatch
from .parallel import Parallel
from .util import register_module

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
def _repeat_concat(z: B.Numeric):
    return z


@_dispatch
def _repeat_concat(zs: Parallel):
    return _repeat_concat(*zs)


@_dispatch
def _repeat_concat(z0: B.Numeric, *zs: B.Numeric):
    # Broadcast the data dimensions before concatenating.
    zs = (z0,) + zs
    # Broadcast the data dimensions.
    shapes = [list(B.shape(z)) for z in zs]
    shape_n = max(shape[2] for shape in shapes)
    for shape in shapes:
        shape[2] = shape_n
    zs = [B.broadcast_to(z, *shape) for (z, shape) in zip(zs, shapes)]
    return B.concat(*zs, axis=1)


@register_module
class Materialise:
    """Materialise an aggregate encoding.

    Args:
        agg_x (function, optional): Aggregation of the inputs. Defaults to merging the
            inputs.
        agg_y (function, optional): Aggregation of the outputs. Defaults to
            concatenating the outputs.
        broadcast_y (bool, optional): Broadcast the data dimensions of the outputs
            before performing `agg_y`. Defaults to `True`.

    Attributes:
        agg_x (function): Aggregation of the inputs.
        agg_y (function): Aggregation of the outputs.
        broadcast_y (bool): Broadcast the data dimensions of the outputs before
            performing `agg_y`.
    """

    def __init__(self, agg_x=_merge, agg_y=_repeat_concat, broadcast_y=True):
        self.agg_x = agg_x
        self.agg_y = agg_y
        self.broadcast_y = broadcast_y


@_dispatch
def code(coder: Materialise, xz, z, x, **kw_args):
    return coder.agg_x(xz), coder.agg_y(z)

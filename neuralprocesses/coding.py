import lab as B
import matrix  # noqa
from plum import Union

from . import _dispatch
from .parallel import Parallel
from .util import register_module, data_dims
from .dist import Dirac, AbstractMultiOutputDistribution

__all__ = [
    "code",
    "code_track",
    "recode",
    "recode_stochastic",
    "Materialise",
]


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
def code_track(coder, xz, z, x, **kw_args):
    """Perform a coding operation whilst tracking the sequence of target inputs, also
    called the history. This history can be used to perform the coding operation again
    at that sequence of target inputs exactly.

    Args:
        coder (coder): Coder.
        xz (input): Current inputs corresponding to current encoding.
        z (tensor): Current encoding.
        x (input): Desired inputs.

    Returns:
        input: Input of encoding.
        tensor: Encoding.
        list: History.
    """
    return code_track(coder, xz, z, x, [], **kw_args)


@_dispatch
def code_track(coder, xz, z, x, h, **kw_args):
    xz, z = code(coder, xz, z, x, **kw_args)
    return xz, z, h + [x]


@_dispatch
def recode(coder, xz, z, h, **kw_args):
    """Perform a coding operation at an earlier recorded sequence of target inputs,
    called the history.

    Args:
        coder (coder): Coder.
        xz (input): Current inputs corresponding to current encoding.
        z (tensor): Current encoding.
        h (list): Target history.

    Returns:
        input: Input of encoding.
        tensor: Encoding.
        list: Remainder of the target history.
    """
    xz, z = code(coder, xz, z, h[0], **kw_args)
    return xz, z, h[1:]


@_dispatch
def code_track(coder, xz, z, x, h, **kw_args):
    xz, z = code(coder, xz, z, x, **kw_args)
    return xz, z, h + [x]


@_dispatch
def recode_stochastic(coders: Parallel, codings: Parallel, xc, yc, h, **kw_args):
    """In an existing aggregate coding `codings`, recode the codings that are not
    :class:`.dist.Dirac`s for a new context set.

    Args:
        coders (:class:`.parallel.Parallel`): Coders that producing the codings.
        codings (:class:`.parallel.Parallel`): Codings.
        xc (tensor): Inputs of new context set.
        yc (tensor): Outputs of new context set.
        h (list): History.

    Returns:
        :class:`.parallel.Parallel`: Updated coding.
    """
    return Parallel(
        *(
            recode_stochastic(coder, coding, xc, yc, hi, **kw_args)
            for (coder, coding, hi) in zip(coders, codings, h[0])
        )
    )


@_dispatch
def recode_stochastic(coder, coding: Dirac, xc, yc, h, **kw_args):
    # Do not recode `Dirac`s.
    return coding


# If the coding is aggregate, it can still contain `Dirac`s, so we need to be careful.


@_dispatch
def recode_stochastic(coder, coding, xc, yc, h, **kw_args):
    # Do not recode `Dirac`s.
    return _choose(recode(coder, xc, yc, h, **kw_args)[1], coding)


@_dispatch
def _choose(new: Parallel, old: Parallel):
    return Parallel(*(_choose(x, y) for x, y in zip(new, old)))


@_dispatch
def _choose(new: Dirac, old: Dirac):
    # Do not recode `Dirac`s.
    return old


@_dispatch
def _choose(new: AbstractMultiOutputDistribution, old: AbstractMultiOutputDistribution):
    # Do recode other distributions.
    return new


@register_module
class Materialise:
    """Materialise an aggregate encoding."""


@_dispatch
def code(coder: Materialise, xz, z, x, **kw_args):
    return _merge(xz), _repeat_concat(xz, z)


@_dispatch
def _merge(z: Union[B.Numeric, None]):
    return z


@_dispatch
def _merge(zs: Parallel):
    return _merge(*zs)


@_dispatch
def _merge(z0: Union[B.Numeric, None], *zs: Union[B.Numeric, None]):
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
    # Some may be batched, but not all.
    rank = max([B.rank(z) for z in zs])
    zs = [B.expand_dims(z, axis=0) if B.rank(z) < rank else z for z in zs]
    # Broadcast the data dimensions. There are `dims` many of them, so perform a loop.
    # Also incoporate the first dimension, because that might be a batch dimension.
    shapes = [list(B.shape(z)) for z in zs]
    for i in [0] + list(range(rank - 1, rank - 1 - dims, -1)):
        shape_n = max(shape[i] for shape in shapes)
        for shape in shapes:
            shape[i] = shape_n
    zs = [B.broadcast_to(z, *shape) for (z, shape) in zip(zs, shapes)]
    return B.concat(*zs, axis=-1 - dims)

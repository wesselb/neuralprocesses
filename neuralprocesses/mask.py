from typing import Tuple, Union

import lab as B
from lab.util import resolve_axis

from . import _dispatch

__all__ = ["Masked", "mask_context", "merge_contexts"]


class Masked:
    """A masked output.

    Args:
        y (tensor): Output to mask. The masked values can have any non-NaN value, but
            they cannot be NaN!
        mask (tensor): A mask consisting of zeros and ones and just one channel.

    Attributes:
        y (tensor): Masked output.
        mask (tensor): A mask consisting of zeros and ones and just one channel.
    """

    def __init__(self, y, mask):
        self.y = y
        self.mask = mask


@B.to_active_device.dispatch
def to_active_device(masked: Masked):
    return Masked(B.to_active_device(masked.y), B.to_active_device(masked.mask))


@_dispatch
def _pad_zeros(x: B.Numeric, up_to: int, axis: int):
    axis = resolve_axis(x, axis)
    shape = list(B.shape(x))
    shape[axis] = up_to - shape[axis]
    with B.on_device(x):
        return B.concat(x, B.zeros(B.dtype(x), *shape), axis=axis)


def _ceil_to_closest_multiple(n, m):
    d, r = divmod(n, m)
    # If `n` is zero, then we must also round up.
    if n == 0 or r > 0:
        return (d + 1) * m
    else:
        return d * m


@_dispatch
def _determine_ns(xc: tuple, multiple: Union[int, tuple]):
    ns = [B.shape(xci, 2) for xci in xc]

    if not isinstance(multiple, tuple):
        multiple = (multiple,) * len(ns)

    # Ceil to the closest multiple of `multiple`.
    ns = [_ceil_to_closest_multiple(n, m) for n, m in zip(ns, multiple)]

    return ns


@_dispatch
def mask_context(xc: tuple, yc: B.Numeric, multiple=1):
    """Mask a context set.

    Args:
        xc (input): Context inputs.
        yc (tensor): Context outputs.
        multiple (int or tuple[int], optional): Pad with zeros until the number of
            context points is a multiple of this number. Defaults to 1.

    Returns:
        tuple[input, :class:`.Masked`]: Masked context set with zeros appended.
    """
    ns = _determine_ns(xc, multiple)

    # Construct the mask, not yet of the final size.
    with B.on_device(yc):
        mask = B.ones(yc)

    # Pad everything with zeros to get the desired size.
    xc = tuple(_pad_zeros(xci, n, 2) for xci, n in zip(xc, ns))
    for i, n in enumerate(ns):
        yc = _pad_zeros(yc, n, 2 + i)
        mask = _pad_zeros(mask, n, 2 + i)

    return xc, Masked(yc, mask)


@_dispatch
def mask_context(xc: B.Numeric, yc: B.Numeric, **kw_args):
    xc, yc = mask_context((xc,), yc, **kw_args)  # Pack input.
    return xc[0], yc  # Unpack input.


@_dispatch
def merge_contexts(*contexts: Tuple[tuple, B.Numeric], multiple=1):
    """Merge context sets.

    Args:
        *contexts (tuple[input, tensor]): Contexts to merge.
        multiple (int or tuple[int], optional): Pad with zeros until the number of
            context points is a multiple of this number. Defaults to 1.

    Returns:
        tuple[input, :class:`.Masked`]: Merged context set.
    """
    ns = tuple(map(max, zip(*(_determine_ns(xc, multiple) for xc, _ in contexts))))
    xcs, ycs = zip(*(mask_context(*context, multiple=ns) for context in contexts))
    ycs, masks = zip(*((yc.y, yc.mask) for yc in ycs))

    return (
        tuple(B.concat(*xcsi, axis=0) for xcsi in zip(*xcs)),
        Masked(B.concat(*ycs, axis=0), B.concat(*masks, axis=0)),
    )


@_dispatch
def merge_contexts(*contexts: Tuple[B.Numeric, B.Numeric], **kw_args):
    xcs, ycs = zip(*contexts)
    xcs = tuple((xc,) for xc in xcs)  # Pack inputs.
    xc, yc = merge_contexts(*zip(xcs, ycs), **kw_args)
    return xc[0], yc  # Unpack inputs.

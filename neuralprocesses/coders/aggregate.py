from typing import Union

import lab as B
import numpy as np
from plum import convert

from .. import _dispatch
from ..aggregate import Aggregate, AggregateInput
from ..datadims import data_dims
from ..util import (
    merge_dimensions,
    register_composite_coder,
    register_module,
    select,
    split,
    split_dimension,
)

__all__ = [
    "RepeatForAggregateInputs",
    "SelectFromChannels",
    "RepeatForAggregateInputPairs",
    "SelectFromDenseCovarianceChannels",
]


@register_composite_coder
@register_module
class RepeatForAggregateInputs:
    """If the coder `coder` encounters an aggregate of target inputs, perform the
    coding operation for every element in the aggregate with the keyword argument
    `select_channel` set to the index of the particular output selected in the element
    of the aggregate of inputs.

    Args:
        coder (coder): Coder.
    """

    def __init__(self, coder):
        self.coder = coder


@_dispatch
def code(coder: RepeatForAggregateInputs, xz, z, x, **kw_args):
    return code(coder.coder, xz, z, x, **kw_args)


@_dispatch
def code_track(coder: RepeatForAggregateInputs, xz, z, x, h, **kw_args):
    return code_track(coder.coder, xz, z, x, h + [x], **kw_args)


@_dispatch
def recode(coder: RepeatForAggregateInputs, xz, z, h, **kw_args):
    # We need to dispatch on the type of `x`, which is hidden in the `recode` signature.
    # Hence, we introduce the additional method `_recode` which exposes the first
    # element of the history.
    return _recode(coder, xz, z, h[0], h[1:], **kw_args)


@_dispatch
def _recode(
    coder: RepeatForAggregateInputs,
    xz: Union[B.Numeric, tuple, None],
    z,
    x: Union[B.Numeric, tuple, None],
    h,
    **kw_args,
):
    return recode(coder.coder, xz, z, h, **kw_args)


@_dispatch
def code(coder: RepeatForAggregateInputs, xz, z, x: AggregateInput, **kw_args):
    xzs, zs = [], []
    for xi, i in x:
        xzi, zi = code(coder.coder, xz, z, xi, select_channel=i, **kw_args)
        xzs.append((xzi, i))
        zs.append(zi)
    return AggregateInput(*xzs), Aggregate(*zs)


@_dispatch
def code_track(coder: RepeatForAggregateInputs, xz, z, x: AggregateInput, h, **kw_args):
    h = h + [x]
    xzs, zs = [], []
    for xi, i in x:
        xzi, zi, h = code_track(coder.coder, xz, z, xi, h, select_channel=i, **kw_args)
        xzs.append((xzi, i))
        zs.append(zi)
    return AggregateInput(*xzs), Aggregate(*zs), h


@_dispatch
def _recode(coder: RepeatForAggregateInputs, xz, z, x: AggregateInput, h, **kw_args):
    xzs, zs = [], []
    for xi, i in x:
        xzi, zi, h = recode(coder.coder, xz, z, h, select_channel=i, **kw_args)
        xzs.append((xzi, i))
        zs.append(zi)
    return AggregateInput(*xzs), Aggregate(*zs), h


@_dispatch
def code(
    coder: RepeatForAggregateInputs,
    xz: AggregateInput,
    z: Aggregate,
    x: AggregateInput,
    **kw_args,
):
    xzs, zs = [], []
    for (xzi, _), zi, (xi, i) in zip(xz, z, x):
        xzi, zi = code(coder.coder, xzi, zi, xi, select_channel=i, **kw_args)
        xzs.append((xzi, i))
        zs.append(zi)
    return AggregateInput(*xzs), Aggregate(*zs)


@_dispatch
def code_track(
    coder: RepeatForAggregateInputs,
    xz: AggregateInput,
    z: Aggregate,
    x: AggregateInput,
    h,
    **kw_args,
):
    h = h + [x]
    xzs, zs = [], []
    for (xzi, _), zi, (xi, i) in zip(xz, z, x):
        xzi, zi, h = code_track(
            coder.coder, xzi, zi, xi, h, select_channel=i, **kw_args
        )
        xzs.append((xzi, i))
        zs.append(zi)
    return AggregateInput(*xzs), Aggregate(*zs), h


@_dispatch
def _recode(
    coder: RepeatForAggregateInputs,
    xz: AggregateInput,
    z: Aggregate,
    x: AggregateInput,
    h,
    **kw_args,
):
    xzs, zs = [], []
    for (xzi, _), zi, (xi, i) in zip(xz, z, x):
        xzi, zi, h = recode(coder.coder, xzi, zi, xi, h, select_channel=i, **kw_args)
        xzs.append((xzi, i))
        zs.append(zi)
    return AggregateInput(*xzs), Aggregate(*zs), h


@register_module
class SelectFromChannels:
    """Select an output channel within a :class:`.RepeatForAggregateInputs`.

    The channels dimension is supposed to be a concatenation of multiple blocks of
    channels, where selecting the `i`th output is performed by selecting the `i`th
    element of every block. The elements of `sizes` specify the lengths of the blocks.

    If an element of `sizes` is a tuple, then it is assumed that the channels of that
    block must be further reshaped into that tuple. The selection will then happen on
    the _last_ unpacked dimension.

    Args:
        *sizes (int or tuple[int]): Specification of the channel blocks.
    """

    def __init__(self, size0, *sizes):
        self.sizes = tuple(convert(s, tuple) for s in (size0,) + sizes)


@_dispatch
def code(
    coder: SelectFromChannels,
    xz,
    z,
    x,
    select_channel=None,
    **kw_args,
):
    if select_channel is not None:
        d = data_dims(xz)
        zs = split(z, tuple(map(np.prod, coder.sizes)), -d - 1)
        zs_selected = []
        for zi, si in zip(zs, coder.sizes):
            zi = split_dimension(zi, -d - 1, si)
            zi = select(zi, select_channel, -d - 1)
            zi = merge_dimensions(zi, -d - 1, si[:-1] + (1,))
            zs_selected.append(zi)
        z = B.concat(*zs_selected, axis=-d - 1)

    return xz, z


@register_composite_coder
@register_module
class RepeatForAggregateInputPairs:
    """If the coder `coder` encounters an aggregate of target inputs, perform the
    coding operation for every pair of elements in the aggregate with the keyword
    arguments `select_channel_i` and `select_channel_j` set to the indices of the
    particular outputs selected in the pair of elements of the aggregate of inputs.

    Args:
        coder (coder): Coder.
    """

    def __init__(self, coder):
        self.coder = coder


@_dispatch
def code(coder: RepeatForAggregateInputPairs, xz, z, x, **kw_args):
    return code(coder.coder, xz, z, x, **kw_args)


@_dispatch
def code(
    coder: RepeatForAggregateInputPairs,
    xz,
    z,
    x: AggregateInput,
    **kw_args,
):
    xz_agg = []
    z_agg = []
    for xi, i in x:
        xzi_agg = []
        zi_agg = []
        for xj, j in x:
            xij = _aggpairs_combine(xi, xj)
            xzij, zij = code(
                coder.coder,
                xz,
                z,
                xij,
                select_channel_i=i,
                select_channel_j=j,
                **kw_args,
            )
            xzi_agg.append((xij, j))
            zi_agg.append(zij)
        xz_agg.append((AggregateInput(*xzi_agg), i))
        z_agg.append(Aggregate(*zi_agg))
    return AggregateInput(*xz_agg), Aggregate(*z_agg)


@_dispatch
def _aggpairs_combine(xi: tuple, xj: tuple):
    ni = len(xi) // 2
    nj = len(xj) // 2
    return xi[:ni] + xj[nj:]


@register_module
class SelectFromDenseCovarianceChannels:
    """Select a pair of output channels from a dense covariance within a
    :class:`RepeatForAggregateInputPairs`."""


@_dispatch
def code(
    coder: SelectFromDenseCovarianceChannels,
    xz,
    z,
    x,
    select_channel_i=None,
    select_channel_j=None,
    **kw_args,
):
    if select_channel_i is not None:
        d = data_dims(xz) // 2
        z = select(z, select_channel_i, 2 * (-d - 1))
        z = select(z, select_channel_j, 1 * (-d - 1))
    return xz, z

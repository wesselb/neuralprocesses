import lab as B

from .. import _dispatch
from ..aggregate import Aggregate, AggregateTargets
from ..util import register_module

__all__ = ["AggregateTargetsCoder", "ConcatenateAggregate"]


@register_module
class AggregateTargetsCoder:
    """If the coder `coder`, which is assumed to be completing, encounters an aggregate
    of target inputs, perform the coding operation for every element in the aggregate.
    Moreover, for every element in the aggregate, after running `coder`, also run
    the coder `selecting_coder` with the keyword argument `select_channel` set to the
    index of the particular output selected in the element of the aggregate of
    target inputs.

    Args:
        coder (coder): Coder.
        selecting_coder (coder, optional): Coder run with keyword argument
            `select_channel` set.
    """

    def __init__(self, coder, selecting_coder=lambda x: x):
        self.coder = coder
        self.selecting_coder = selecting_coder


@_dispatch
def code(coder: AggregateTargetsCoder, xz, z, x, **kw_args):
    xz, z = code(coder.coder, xz, z, x, **kw_args)
    xz, z = code(coder.selecting_coder, xz, z, x, **kw_args)
    return xz, z


@_dispatch
def code_track(coder: AggregateTargetsCoder, xz, z, x, h, **kw_args):
    h = h + [x]
    xz, z, h = code_track(coder.coder, xz, z, x, h, **kw_args)
    xz, z, h = code_track(coder.selecting_coder, xz, z, x, h, **kw_args)
    return xz, z, h


@_dispatch
def recode(coder: AggregateTargetsCoder, xz, z, h, **kw_args):
    # We need to dispatch on the type of `x`, which is hidden in the `recode` signature.
    # Hence, we introduce the additional method `_recode` which exposes the first
    # element of the history.
    return _recode(coder, xz, z, h[0], h[1:], **kw_args)


@_dispatch
def _recode(coder: AggregateTargetsCoder, xz, z, x, h, **kw_args):
    xz, z, h = recode(coder.coder, xz, z, h, **kw_args)
    xz, z, h = recode(coder.selecting_coder, xz, z, h, **kw_args)
    return xz, z, h


@_dispatch
def code(coder: AggregateTargetsCoder, xz, z, x: AggregateTargets, **kw_args):
    xzs, zs = [], []
    for xi, i in x:
        xzi, zi = code(coder.coder, xz, z, xi, **kw_args)
        xzi, zi = code(coder.selecting_coder, xzi, zi, xi, select_channel=i, **kw_args)
        xzs.append(xzi)
        zs.append(zi)
    return AggregateTargets(*((xzi, i) for xzi, (_, i) in zip(xzs, x))), Aggregate(*zs)


@_dispatch
def code_track(coder: AggregateTargetsCoder, xz, z, x: AggregateTargets, h, **kw_args):
    h = h + [x]
    xzs, zs = [], []
    for xi, i in x:
        xzi, zi, h = code_track(coder.coder, xz, z, xi, h, **kw_args)
        xzi, zi, h = code_track(
            coder.selecting_coder,
            xzi,
            zi,
            xi,
            h,
            select_channel=i,
            **kw_args,
        )
        xzs.append(xzi)
        zs.append(zi)
    return (
        AggregateTargets(*((xzi, i) for xzi, (_, i) in zip(xzs, x))),
        Aggregate(*zs),
        h,
    )


@_dispatch
def _recode(coder: AggregateTargetsCoder, xz, z, x: AggregateTargets, h, **kw_args):
    xzs, zs = [], []
    for xi, i in x:
        xzi, zi, h = recode(coder.coder, xz, z, h, **kw_args)
        xzi, zi, h = recode(
            coder.selecting_coder,
            xzi,
            zi,
            h,
            select_channel=i,
            **kw_args,
        )
        xzs.append(xzi)
        zs.append(zi)
    return (
        AggregateTargets(*((xzi, i) for xzi, (_, i) in zip(xzs, x))),
        Aggregate(*zs),
        h,
    )


@_dispatch
def code(
    coder: AggregateTargetsCoder,
    xz: AggregateTargets,
    z: Aggregate,
    x: AggregateTargets,
    **kw_args,
):
    xzs, zs = [], []
    for (xzi, _), zi, (xi, i) in zip(xz, z, x):
        xzi, zi = code(coder.coder, xzi, zi, xi, **kw_args)
        xzi, zi = code(coder.selecting_coder, xzi, zi, xi, select_channel=i, **kw_args)
        xzs.append(xzi)
        zs.append(zi)
    return AggregateTargets(*((xzi, i) for xzi, (_, i) in zip(xzs, x))), Aggregate(*zs)


@_dispatch
def code_track(
    coder: AggregateTargetsCoder,
    xz: AggregateTargets,
    z: Aggregate,
    x: AggregateTargets,
    h,
    **kw_args,
):
    h = h + [x]
    xzs, zs = [], []
    for (xzi, _), zi, (xi, i) in zip(xz, z, x):
        xzi, zi, h = code_track(coder.coder, xzi, zi, xi, h, **kw_args)
        xzi, zi, h = code_track(
            coder.selecting_coder,
            xzi,
            zi,
            xi,
            h,
            select_channel=i,
            **kw_args,
        )
        xzs.append(xzi)
        zs.append(zi)
    return (
        AggregateTargets(*((xzi, i) for xzi, (_, i) in zip(xzs, x))),
        Aggregate(*zs),
        h,
    )


@_dispatch
def _recode(
    coder: AggregateTargetsCoder,
    xz: AggregateTargets,
    z: Aggregate,
    x: AggregateTargets,
    h,
    **kw_args,
):
    xzs, zs = [], []
    for (xzi, _), zi, (xi, i) in zip(xz, z, x):
        xzi, zi, h = recode(coder.coder, xzi, zi, h, **kw_args)
        xzi, zi, h = recode(
            coder.selecting_coder,
            xzi,
            zi,
            h,
            select_channel=i,
            **kw_args,
        )
        xzs.append(xzi)
        zs.append(zi)
    return (
        AggregateTargets(*((xzi, i) for xzi, (_, i) in zip(xzs, x))),
        Aggregate(*zs),
        h,
    )


@register_module
class ConcatenateAggregate:
    """If the encoding is aggregate, concatenate it using `B.concat`; otherwise, do
    nothing."""


@_dispatch
def code(coder: ConcatenateAggregate, xz, z, x, **kw_args):
    return xz, z


@_dispatch
def code(coder: ConcatenateAggregate, xz, z: Aggregate, x, **kw_args):
    return xz, B.concat(*z)

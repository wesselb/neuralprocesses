import lab as B
import matrix  # noqa

from .. import _dispatch
from ..aggregate import AggregateInput
from ..util import register_module, register_composite_coder
from ..parallel import Parallel

__all__ = ["MapDiagonal"]


@register_composite_coder
@register_module
class MapDiagonal:
    """Map to the diagonal of the squared space.

    Args:
        coder (coder): Coder to apply the mapped values to.

    Attributes:
        coder (coder): Coder to apply the mapped values to.
    """

    def __init__(self, coder):
        self.coder = coder


@_dispatch
def code(coder: MapDiagonal, xz, z, x, **kw_args):
    x, d = _mapdiagonal_duplicate_target(x)
    # The encoding might already be on the diagonal. Therefore, only duplicate the
    # inputs if the dimensionalities don't line up.
    xz = _mapdiagonal_possibly_duplicate_context(xz, d)
    return code(coder.coder, xz, z, x, **kw_args)


@_dispatch
def code_track(coder: MapDiagonal, xz, z, x, h, **kw_args):
    x, d = _mapdiagonal_duplicate_target(x)
    xz = _mapdiagonal_possibly_duplicate_context(xz, d)
    return code_track(coder.coder, xz, z, x, h + [(x, d)], **kw_args)


@_dispatch
def recode(coder: MapDiagonal, xz, z, h, **kw_args):
    (_, d), h = h[0], h[1:]
    xz = _mapdiagonal_possibly_duplicate_context(xz, d)
    return recode(coder.coder, xz, z, h, **kw_args)


@_dispatch
def _mapdiagonal_duplicate_target(x: B.Numeric):
    return (x, x), 2


@_dispatch
def _mapdiagonal_duplicate_target(x: AggregateInput):
    xis, ds = zip(*(_mapdiagonal_duplicate_target(xi) for xi, _ in x))
    if not all([d == ds[0] for d in ds[1:]]):
        raise NotImplementedError("All data dimensionalities must be equal.")
    else:
        d = ds[0]
    return AggregateInput(*((xi, i) for xi, (_, i) in zip(xis, x))), d


@_dispatch
def _mapdiagonal_possibly_duplicate_context(xz: B.Numeric, d: B.Int):
    if B.shape(xz, -2) != d:
        return B.concat(xz, xz, axis=-2)
    else:
        return xz


@_dispatch
def _mapdiagonal_possibly_duplicate_context(xz: tuple, d: B.Int):
    if len(xz) != d:
        return xz * 2
    else:
        return xz


@_dispatch
def _mapdiagonal_possibly_duplicate_context(xz: Parallel, d: B.Int):
    return Parallel(*(_mapdiagonal_possibly_duplicate_context(xzi, d) for xzi in xz))

import lab as B

from .. import _dispatch
from ..datadims import data_dims
from ..parallel import Parallel
from ..util import register_module, split

__all__ = ["Identity", "Splitter", "RestructureParallel"]


@register_module
class Identity:
    """Identity coder."""


@_dispatch
def code(coder: Identity, xz, z, x, **kw_args):
    return xz, z


@register_module
class Splitter:
    """Split a tensor into multiple tensors.

    Args:
        *sizes (int): Size of every split

    Attributes:
        sizes (tuple[int]): Size of every split
    """

    def __init__(self, size0, *sizes):
        self.sizes = (size0,) + sizes


@_dispatch
def code(coder: Splitter, xz, z: B.Numeric, x, **kw_args):
    d = data_dims(xz)
    return xz, Parallel(*split(z, coder.sizes, -d - 1))


@register_module
class RestructureParallel:
    """Restructure a parallel of things.

    Args:
        current (tuple): Current structure.
        new (tuple): New structure.

    Attributes:
        current (tuple): Current structure.
        new (tuple): New structure.
    """

    def __init__(self, current, new):
        self.current = current
        self.new = new


@_dispatch
def code(coder: RestructureParallel, xz: Parallel, z: Parallel, x, **kw_args):
    return (
        _restructure(xz, coder.current, coder.new),
        _restructure(z, coder.current, coder.new),
    )


@_dispatch
def _restructure(p: Parallel, current: tuple, new: tuple):
    element_map = {}
    _restructure_assign(element_map, p, current)
    return _restructure_create(element_map, new)


@_dispatch
def _restructure_assign(element_map: dict, obj, i):
    element_map[i] = obj


@_dispatch
def _restructure_assign(element_map: dict, p: Parallel, x: tuple):
    if len(p) != len(x):
        raise RuntimeError("Parallel does not match structure.")
    for pi, xi in zip(p, x):
        _restructure_assign(element_map, pi, xi)


@_dispatch
def _restructure_create(element_map, i):
    return element_map[i]


@_dispatch
def _restructure_create(element_map, x: tuple):
    return Parallel(*(_restructure_create(element_map, xi) for xi in x))

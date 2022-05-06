import lab as B

from .. import _dispatch
from ..datadims import data_dims
from ..parallel import Parallel
from ..util import register_module, split

__all__ = ["Splitter"]


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

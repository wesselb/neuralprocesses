import lab as B

from .. import _dispatch
from ..datadims import data_dims
from ..parallel import Parallel
from ..util import register_module

__all__ = ["Fuse"]


@register_module
class Fuse:
    """In a parallel of two things, interpolate the first to the inputs of the second,
    and concatenate the result to the second.

    Args:
        set_conv (coder): Set conv that should perform the interpolation.

    Attributes:
        set_conv (coder): Set conv that should perform the interpolation.
    """

    def __init__(self, set_conv):
        self.set_conv = set_conv


@_dispatch
def code(coder: Fuse, xz: Parallel, z: Parallel, x, **kw_args):
    xz1, xz2 = xz
    z1, z2 = z
    _, z1 = code(coder.set_conv, xz1, z1, xz2, **kw_args)
    return xz2, B.concat(z1, z2, axis=-1 - data_dims(xz2))

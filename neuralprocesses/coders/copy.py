from .. import _dispatch
from ..parallel import Parallel
from ..util import register_module

__all__ = ["Copy"]


@register_module
class Copy:
    def __init__(self, times):
        self.times = times


@_dispatch
def code(coder: Copy, xz, z, x, **kw_args):
    return Parallel(*((xz,) * coder.times)), Parallel(*((z,) * coder.times))

import lab as B
import matrix  # noqa

from .. import _dispatch
from ..util import register_module

__all__ = ["MapDiagonal"]


@register_module
class MapDiagonal:
    """Map to the diagonal of the squared space.

    Args:
        coder (coder): Coder to apply the mapped values to.
        map_encoding (bool, optional): Also map the encoding to the diagonal. Set
            this to `False` if the encoder had already been mapped to the diagonal.
            Defaults to `True`.

    Attributes:
        coder (function): Coder to apply the mapped values to.
        map_encoding (bool): Map the encoding to the diagonal.
    """

    def __init__(self, coder, map_encoding=True):
        self.coder = coder
        self.map_encoding = map_encoding


@_dispatch
def code(coder: MapDiagonal, xz, z, x: B.Numeric, **kw_args):
    if coder.map_encoding:
        xz = B.concat(xz, xz, axis=-2)
    return code(coder.coder, xz, z, (x, x), **kw_args)

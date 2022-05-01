import lab as B
import matrix  # noqa

from .. import _dispatch
from ..util import register_module

__all__ = ["DeepSet"]


@register_module
class DeepSet:
    """Deep set.

    Args:
        phi (object): Pre-aggregation function.
        agg (object, optional): Aggregation function. Defaults to summing.

    Attributes:
        phi (object): Pre-aggregation function.
        agg (object): Aggregation function.
    """

    def __init__(
        self,
        phi,
        agg=lambda x: B.sum(x, axis=-1, squeeze=False),
    ):
        self.phi = phi
        self.agg = agg


@_dispatch
def code(coder: DeepSet, xz: B.Numeric, z: B.Numeric, x, **kw_args):
    z = B.concat(xz, z, axis=-2)
    z = coder.phi(z)
    z = coder.agg(z)  # This aggregates over the data dimension.
    return None, z

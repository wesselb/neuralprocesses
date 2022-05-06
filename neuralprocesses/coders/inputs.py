import lab as B
import matrix  # noqa

from .. import _dispatch
from ..aggregate import Aggregate, AggregateInput
from ..util import register_module

__all__ = ["InputsCoder"]


@register_module
class InputsCoder:
    """Encode with the target inputs."""


@_dispatch
def code(coder: InputsCoder, xz, z, x: B.Numeric, **kw_args):
    return x, x


@_dispatch
def code(coder: InputsCoder, xz, z, x: AggregateInput, **kw_args):
    return x, Aggregate(*(xi for xi, i in x))

import matrix  # noqa

from .. import _dispatch
from ..util import register_module

__all__ = ["FunctionalCoder"]


@register_module
class FunctionalCoder:
    """A coder that codes to a discretisation for a functional representation.

    Args:
        disc (:class:`.discretisation.AbstractDiscretisation`): Discretisation.
        coder (coder): Coder.

    Attributes:
        disc (:class:`.discretisation.AbstractDiscretisation`): Discretisation.
        coder (coder): Coder.
    """

    def __init__(self, disc, coder):
        self.disc = disc
        self.coder = coder


@_dispatch
def code(coder: FunctionalCoder, xz, z, x, **kw_args):
    x = coder.disc(xz, x, **kw_args)
    return code(coder.coder, xz, z, x, **kw_args)


@_dispatch
def code_track(coder: FunctionalCoder, xz, z, x, h, **kw_args):
    x = coder.disc(xz, x, **kw_args)
    return code_track(coder.coder, xz, z, x, h + [x], **kw_args)


@_dispatch
def recode(coder: FunctionalCoder, xz, z, h, **kw_args):
    return recode(coder.coder, xz, z, h[1:], **kw_args)

import matrix  # noqa
from plum import convert

from .. import _dispatch
from ..util import register_module, register_composite_coder

__all__ = ["FunctionalCoder"]


@register_composite_coder
@register_module
class FunctionalCoder:
    """A coder that codes to a discretisation for a functional representation.

    Args:
        disc (:class:`.discretisation.AbstractDiscretisation`): Discretisation.
        coder (coder): Coder.
        target (function, optional): Function which takes in the inputs of the current
            encoding and the desired inputs and which returns a tuple containing the
            inputs to span the discretisation over.

    Attributes:
        disc (:class:`.discretisation.AbstractDiscretisation`): Discretisation.
        coder (coder): Coder.
        target (function): Function which takes in the inputs of the current encoding
            and the desired inputs and which returns a tuple containing the inputs to
            span the discretisation over.
    """

    def __init__(self, disc, coder, target=lambda xc, xt: (xc, xt)):
        self.disc = disc
        self.coder = coder
        self.target = target


@_dispatch
def code(coder: FunctionalCoder, xz, z, x, **kw_args):
    x = coder.disc(*coder.target(xz, x), **kw_args)
    return code(coder.coder, xz, z, x, **kw_args)


@_dispatch
def code_track(coder: FunctionalCoder, xz, z, x, h, **kw_args):
    x = coder.disc(*coder.target(xz, x), **kw_args)
    return code_track(coder.coder, xz, z, x, h + [x], **kw_args)


@_dispatch
def recode(coder: FunctionalCoder, xz, z, h, **kw_args):
    return recode(coder.coder, xz, z, h[1:], **kw_args)

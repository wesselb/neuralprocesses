import lab as B

from .. import _dispatch
from ..augment import AugmentedInput
from ..datadims import data_dims
from ..materialise import _repeat_concat
from ..util import register_composite_coder, register_module

__all__ = ["Augment", "AssertNoAugmentation"]


@register_composite_coder
@register_module
class Augment:
    """Concatenate the augmentation of the input to the encoding, and remove any
    augmentation of target inputs.

    Args:
        coder (coder): Coder to run after the augmentation.

    Attributes:
        coder (coder): Coder to run after the augmentation.
    """

    def __init__(self, coder):
        self.coder = coder


@_dispatch
def code(
    coder: Augment,
    xz,
    z,
    x,
    **kw_args,
):
    xz, z = _augment(xz, z)
    x = _augment(x)
    return code(coder.coder, xz, z, x, **kw_args)


@_dispatch
def _augment(xz: AugmentedInput, z: B.Numeric):
    return xz.x, _repeat_concat(data_dims(xz), z, xz.augmentation)


@_dispatch
def _augment(xz: AugmentedInput):
    return xz.x


@register_module
class AssertNoAugmentation:
    """Assert no augmentation of the target inputs."""


@_dispatch
def code(coder: AssertNoAugmentation, xz, z, x, **kw_args):
    return xz, z


@_dispatch
def code(coder: AssertNoAugmentation, xz, z, x: AugmentedInput, **kw_args):
    raise AssertionError("Did not expect augmentation of the target inputs.")

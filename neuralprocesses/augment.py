import lab as B

from . import _dispatch
from .util import register_module, data_dims

__all__ = ["AugmentedInput", "Augment"]


class AugmentedInput:
    """An augmented input.

    Args:
        x (input): Input.
        augmentation (object): Augmentation.

    Attributes:
        x (input): Input.
        augmentation (object): Augmentation.
    """

    def __init__(self, x, augmentation):
        self.x = x
        self.augmentation = augmentation


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
    d = data_dims(xz.x)
    return xz.x, B.concat(z, xz.augmentation, axis=-1 - d)


@_dispatch
def _augment(xz: B.Numeric, z: B.Numeric):
    return xz, z


@_dispatch
def _augment(xz: AugmentedInput):
    return xz.x


@_dispatch
def _augment(xz: B.Numeric):
    return xz

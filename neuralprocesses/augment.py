import lab as B

from . import _dispatch
from .util import register_module

__all__ = ["AugmentedInput", "Augment"]


class AugmentedInput:
    def __init__(self, x, augmentation):
        self.x = x
        self.augmentation = augmentation


@register_module
class Augment:
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
    return xz.x, B.concat(z, xz.augmentation, axis=1)


@_dispatch
def _augment(xz: AugmentedInput):
    return xz.x


@_dispatch
def _augment(xz: B.Numeric, z: B.Numeric):
    return xz, z


@_dispatch
def _augment(xz: B.Numeric):
    return xz.x


@_dispatch
def _augment(xz: AugmentedInput, z=None):
    if z is None:
        return xz.x
    else:
        return xz.x, B.concat(z, xz.augmentation, axis=1)

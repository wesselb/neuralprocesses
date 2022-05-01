import lab as B

from ... import _dispatch
from ...datadims import data_dims
from ...util import register_module, batch

__all__ = ["PrependIdentityChannel"]


@register_module
class PrependIdentityChannel:
    """Prepend a density channel to the current encoding."""


@_dispatch
def code(coder: PrependIdentityChannel, xz, z: B.Numeric, x, **kw_args):
    d = data_dims(xz)
    b = batch(z, d + 1)
    with B.on_device(z):
        if d == 2:
            identity_channel = B.diag_construct(B.ones(B.dtype(z), B.shape(z, -1)))
        else:
            raise RuntimeError(
                f"Cannot construct identity channels for encodings of "
                f"dimensionality {d}."
            )
    identity_channel = B.tile(
        B.expand_dims(identity_channel, axis=0, times=len(b) + 1),
        *b,
        1,
        *((1,) * d),
    )
    return xz, B.concat(identity_channel, z, axis=-d - 1)

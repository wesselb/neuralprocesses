from typing import Optional

import lab as B

from ... import _dispatch
from ...datadims import data_dims
from ...mask import Masked
from ...parallel import broadcast_coder_over_parallel
from ...util import register_module, batch

__all__ = [
    "PrependDensityChannel",
    "DivideByFirstChannel",
]


@register_module
class PrependDensityChannel:
    """Prepend a density channel to the current encoding."""


@_dispatch
def code(coder: PrependDensityChannel, xz, z: B.Numeric, x, **kw_args):
    d = data_dims(xz)
    with B.on_device(z):
        density_channel = B.ones(B.dtype(z), *batch(z, d + 1), 1, *B.shape(z)[-d:])
    return xz, B.concat(density_channel, z, axis=-d - 1)


broadcast_coder_over_parallel(PrependDensityChannel)


@_dispatch
def code(coder: PrependDensityChannel, xz, z: Masked, x, **kw_args):
    # Apply mask to density channel _and_ the data channels. Since the mask has
    # only one channel, we can simply pointwise multiply and broadcasting should
    # do the rest for us.
    mask = z.mask
    xz, z = code(coder, xz, z.y, x, **kw_args)
    return xz, (mask * z)


@register_module
class DivideByFirstChannel:
    """Divide by the first channel.

    Args:
        epsilon (float): Value to add to the channel before dividing.

    Attributes:
        epsilon (float): Value to add to the channel before dividing.
    """

    @_dispatch
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon


@_dispatch
def code(
    coder: DivideByFirstChannel,
    xz,
    z: B.Numeric,
    x,
    epsilon: Optional[float] = None,
    **kw_args,
):
    epsilon = epsilon or coder.epsilon
    d = data_dims(xz)
    slice_to_one = (Ellipsis, slice(None, 1, None)) + (slice(None, None, None),) * d
    slice_from_one = (Ellipsis, slice(1, None, None)) + (slice(None, None, None),) * d
    return (
        xz,
        B.concat(
            z[slice_to_one],
            z[slice_from_one] / (z[slice_to_one] + epsilon),
            axis=-d - 1,
        ),
    )


broadcast_coder_over_parallel(DivideByFirstChannel)

from typing import Optional, Union

import lab as B
from plum import isinstance

from ... import _dispatch
from ...datadims import data_dims
from ...mask import Masked
from ...parallel import broadcast_coder_over_parallel
from ...util import batch, register_module

__all__ = [
    "PrependDensityChannel",
    "PrependMultiDensityChannel",
    "DivideByFirstChannel",
    "DivideByFirstHalf",
]


@register_module
class PrependDensityChannel:
    """Prepend a density channel to the current encoding.

    Args:
        multi (bool, optional): Produce a separate density channel for every channel.
            If `False`, produce just one density channel for all channels. Defaults
            to `False`.
    """

    def __init__(self, multi=False):
        self.multi = multi


@register_module
class PrependMultiDensityChannel(PrependDensityChannel):
    """Prepend a separate density channel for every channel to the current encoding."""

    def __init__(self):
        PrependDensityChannel.__init__(self, multi=True)


@_dispatch
def code(coder: PrependDensityChannel, xz, z: B.Numeric, x, **kw_args):
    d = data_dims(xz)
    with B.on_device(z):
        if coder.multi:
            # Produce separate density channels.
            c = B.shape(z, -d - 1)
        else:
            # Produce just one density channel.
            c = 1
        density_channel = B.ones(B.dtype(z), *batch(z, d + 1), c, *B.shape(z)[-d:])
    return xz, B.concat(density_channel, z, axis=-d - 1)


broadcast_coder_over_parallel(PrependDensityChannel)


@_dispatch
def code(coder: PrependDensityChannel, xz, z: Masked, x, **kw_args):
    mask = z.mask
    d = data_dims(xz)
    # Set the missing values to zero by multiplying with the mask. Zeros in the data
    # channel do not affect the encoding.
    return xz, B.concat(z.mask, z.y * z.mask, axis=-d - 1)


@register_module
class DivideByChannels:
    """Divide by the first `n` channels.

    Args:
        spec (int or str): Channels to divide by.
        epsilon (float): Value to add to the channel before dividing.

    Attributes:
        spec (int or str): Channels to divide by.
        epsilon (float): Value to add to the channel before dividing.
    """

    @_dispatch
    def __init__(self, spec: Union[int, str], epsilon: float):
        self.spec = spec
        self.epsilon = epsilon


@register_module
class DivideByFirstChannel(DivideByChannels):
    """Divide by the first channel.

    Args:
        epsilon (float): Value to add to the channel before dividing.

    Attributes:
        epsilon (float): Value to add to the channel before dividing.
    """

    @_dispatch
    def __init__(self, epsilon: float = 1e-8):
        DivideByChannels.__init__(self, 1, epsilon)


@register_module
class DivideByFirstHalf(DivideByChannels):
    """Divide by the first half of channels.

    Args:
        epsilon (float): Value to add to the channels before dividing.

    Attributes:
        epsilon (float): Value to add to the channels before dividing.
    """

    @_dispatch
    def __init__(self, epsilon: float = 1e-8):
        DivideByChannels.__init__(self, "half", epsilon)


@_dispatch
def code(
    coder: DivideByChannels,
    xz,
    z: B.Numeric,
    x,
    epsilon: Optional[float] = None,
    **kw_args,
):
    epsilon = epsilon or coder.epsilon
    d = data_dims(xz)
    if isinstance(coder.spec, B.Int):
        num_divide = coder.spec
    elif coder.spec == "half":
        num_divide = B.shape(z, -d - 1) // 2
    else:
        raise ValueError(f"Unknown specification `{coder.spec}`.")
    slice1 = (Ellipsis, slice(None, num_divide, None)) + (slice(None, None, None),) * d
    slice2 = (Ellipsis, slice(num_divide, None, None)) + (slice(None, None, None),) * d
    return (
        xz,
        B.concat(
            z[slice1],
            z[slice2] / (z[slice1] + epsilon),
            axis=-d - 1,
        ),
    )


broadcast_coder_over_parallel(DivideByChannels)

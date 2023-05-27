import lab as B
import numpy as np
from plum import isinstance

from .util import nps  # noqa


def test_unet_1d(nps):
    unet = nps.UNet(
        dim=1,
        in_channels=3,
        out_channels=4,
        channels=(8, 16, 16, 32, 32, 64),
    )
    n = 2 * 2**unet.num_halving_layers
    z = B.randn(nps.dtype, 2, 3, n)
    assert B.shape(unet(z)) == (2, 4, n)
    assert 40_000 <= nps.num_params(unet) <= 60_000


def test_unet_1d_receptive_field(nps):
    unet = nps.UNet(
        dim=1,
        in_channels=1,
        out_channels=1,
        channels=(3, 5, 7, 5, 3),
        activations=(B.identity,) * 5,
    )
    # Run the model once.
    mult = 2**unet.num_halving_layers
    x = B.zeros(nps.dtype, 1, 1, mult)
    unet(x)
    # Set all weights to one.
    if isinstance(nps.dtype, B.TFDType):
        unet.set_weights(0 * np.array(unet.get_weights(), dtype=object) + 1)
    elif isinstance(nps.dtype, B.TorchDType):
        for p in unet.parameters():
            p.data = p.data * 0 + 1
    else:
        raise RuntimeError("I don't know how to set the weights of the model.")
    for offset in range(unet.receptive_field):
        # Create perturbation.
        x = B.zeros(1, 1, int(10 * unet.receptive_field / mult) * mult)
        x[0, 0, 5 * unet.receptive_field + offset] = 1
        x = B.cast(nps.dtype, x)
        # Check that the computed receptive field is indeed right.
        n = B.sum(B.cast(B.dtype(x), B.abs(B.flatten(unet(x * 0) - unet(x))) > 0))
        assert n == unet.receptive_field


def test_unet_2d(nps):
    unet = nps.UNet(
        dim=2,
        in_channels=3,
        out_channels=4,
        channels=(8, 16, 16, 32, 32, 64),
    )
    n = 2 * 2**unet.num_halving_layers
    z = B.randn(nps.dtype, 2, 3, n, n)
    assert B.shape(unet(z)) == (2, 4, n, n)
    assert 200_000 <= nps.num_params(unet) <= 300_000

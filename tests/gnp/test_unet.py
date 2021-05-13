import lab as B
import torch

import neuralprocesses.gnp as gnp


def test_unet_1d():
    unet = gnp.UNet(
        dimensionality=1,
        in_channels=3,
        out_channels=4,
        channels=(8, 16, 16, 32, 32, 64),
    )
    assert 40_000 <= gnp.num_params(unet) <= 60_000
    n = 2 * 2 ** unet.num_halving_layers
    z = B.randn(torch.float32, 2, 3, n)
    assert B.shape(unet(z)) == (2, 4, n)


def test_unet_2d():
    unet = gnp.UNet(
        dimensionality=2,
        in_channels=3,
        out_channels=4,
        channels=(8, 16, 16, 32, 32, 64),
    )
    assert 200_000 <= gnp.num_params(unet) <= 300_000
    n = 2 * 2 ** unet.num_halving_layers
    z = B.randn(torch.float32, 2, 3, n, n)
    assert B.shape(unet(z)) == (2, 4, n, n)

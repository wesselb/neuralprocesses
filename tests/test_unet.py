import lab as B

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

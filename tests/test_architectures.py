import lab as B
import numpy as np
import pytest

from .util import nps, generate_data  # noqa


@pytest.mark.parametrize("float64", [False, True])
@pytest.mark.parametrize(
    "construct_name, kw_args",
    [
        (model, dict(base_kw_args, dim_x=dim_x, dim_y=dim_y, likelihood=lik))
        for model, base_kw_args in [
            (
                "construct_gnp",
                {
                    "num_basis_functions": 4,
                },
            ),
            (
                "construct_convgnp",
                {
                    "num_basis_functions": 4,
                    "points_per_unit": 16,
                    "unet_channels": (4, 8),
                    "unet_kernels": (5,) * 2,
                    "unet_activations": (B.relu,) * 2,
                    "epsilon": 1e-2,
                },
            ),
            (
                "construct_convgnp",
                {
                    "num_basis_functions": 4,
                    "points_per_unit": 16,
                    "unet_channels": (8, 16),
                    "unet_kernels": (5,) * 2,
                    "unet_activations": (B.relu,) * 2,
                    "unet_resize_convs": False,
                    "epsilon": 1e-2,
                },
            ),
            (
                "construct_convgnp",
                {
                    "num_basis_functions": 4,
                    "points_per_unit": 16,
                    "unet_channels": (8, 16),
                    "unet_kernels": (5,) * 2,
                    "unet_activations": (B.relu,) * 2,
                    "unet_resize_convs": True,
                    "epsilon": 1e-2,
                },
            ),
            (
                "construct_convgnp",
                {
                    "num_basis_functions": 4,
                    "points_per_unit": 16,
                    "conv_arch": "dws",
                    "dws_channels": 8,
                    "dws_layers": 4,
                    "dws_receptive_field": 0.5,
                    "epsilon": 1e-2,
                },
            ),
        ]
        for dim_x in [1, 2]
        for dim_y in [1, 2]
        for lik in ["het", "lowrank", "lowrank-correlated"]
    ],
)
@pytest.mark.flaky(reruns=3)
def test_architectures(nps, float64, construct_name, kw_args):
    if float64:
        nps.dtype = nps.dtype64
    model = getattr(nps, construct_name)(**kw_args, dtype=nps.dtype)
    xc, yc, xt, yt = generate_data(nps, dim_x=kw_args["dim_x"], dim_y=kw_args["dim_y"])
    pred = model(xc, yc, xt, batch_size=2, unused_arg=None)
    objective = B.sum(pred.logpdf(yt))
    # Check that the objective is finite and of the right data type.
    assert np.isfinite(B.to_numpy(objective))
    assert B.dtype(objective) == nps.dtype

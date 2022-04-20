import lab as B
import numpy as np
import pytest

from .util import nps, approx, generate_data  # noqa


def generate_arch_variations(name, **kw_args):
    variations = [
        {
            "unet_channels": (4, 8),
            "unet_kernels": (5,) * 2,
            "unet_activations": (B.relu,) * 2,
        },
        {
            "unet_channels": (8, 16),
            "unet_kernels": (5,) * 2,
            "unet_activations": (B.relu,) * 2,
            "unet_resize_convs": False,
        },
        {
            "unet_channels": (8, 16),
            "unet_kernels": (5,) * 2,
            "unet_activations": (B.relu,) * 2,
            "unet_resize_convs": True,
        },
        {
            "conv_arch": "dws",
            "dws_channels": 8,
            "dws_layers": 4,
            "dws_receptive_field": 0.5,
        },
    ]
    return [(name, dict(config, **kw_args)) for config in variations]


@pytest.mark.parametrize("float64", [False, True])
@pytest.mark.parametrize(
    "construct_name, kw_args",
    [
        (model, dict(base_kw_args, dim_x=dim_x, dim_y=dim_y, likelihood=lik))
        for model, base_kw_args in [
            (
                "construct_gnp",
                {"num_basis_functions": 4},
            ),
            (
                "construct_agnp",
                {"num_basis_functions": 4},
            ),
        ]
        + generate_arch_variations(
            "construct_convgnp",
            points_per_unit=16,
            num_basis_functions=16,
            epsilon=1e-4,
        )
        for dim_x in [1, 2]
        for dim_y in [1, 2]
        for lik in ["het", "lowrank"]
    ]
    + generate_arch_variations(
        "construct_fullconvgnp",
        dim_x=1,
        dim_y=1,
        points_per_unit=16,
        epsilon=1e-4,
    ),
)
@pytest.mark.flaky(reruns=3)
def test_architectures(nps, float64, construct_name, kw_args):
    if float64:
        nps.dtype = nps.dtype64
    model = getattr(nps, construct_name)(**kw_args, dtype=nps.dtype)
    # Generate data.
    xc, yc, xt, yt = generate_data(nps, dim_x=kw_args["dim_x"], dim_y=kw_args["dim_y"])
    model(xc, yc, xt)  # Run the model once to make sure all parameters exist.

    # Perturb all parameters to make sure that the biases aren't initialised to zero. If
    # biases are initialised to zero, then this can give zeros in the output if the
    # input is zero.
    if isinstance(nps.dtype, B.TFDType):
        weights = []
        for p in model.get_weights():
            weights.append(p + 0.01 * B.randn(p))
        model.set_weights(weights)
    elif isinstance(nps.dtype, B.TorchDType):
        for p in model.parameters():
            p.data = p.data + 0.01 * B.randn(p.data)
    else:
        raise RuntimeError("I don't know how to perturb the parameters of the model.")

    # Check passing in a non-empty context set.
    pred = model(xc, yc, xt, batch_size=2, unused_arg=None)
    objective = B.sum(pred.logpdf(yt))
    # Check that the objective is finite and of the right data type.
    assert np.isfinite(B.to_numpy(objective))
    assert B.dtype(objective) == nps.dtype

    # Check that batching works correctly.
    pred2 = model(
        B.reshape(xc, 2, -1, *B.shape(xc, 1, 2)),
        B.reshape(yc, 2, -1, *B.shape(yc, 1, 2)),
        B.reshape(xt, 2, -1, *B.shape(xt, 1, 2)),
        batch_size=2,
        unused_arg=None,
    )
    approx(pred.mean, B.reshape(pred2.mean, -1, *B.shape(pred2.mean, 2, 3)))
    approx(pred.var, B.reshape(pred2.var, -1, *B.shape(pred2.var, 2, 3)))

    # Check passing the edge case of an empty context set.
    if isinstance(xc, B.Numeric):
        xc = xc[:, :, :0]
    elif isinstance(xc, tuple):
        xc = tuple(xci[:, :, :0] for xci in xc)
    else:
        raise RuntimeError("Failed to contruct empty context set.")
    yc = yc[:, :, :0]
    pred = model(xc, yc, xt, batch_size=2, unused_arg=None)
    objective = B.sum(pred.logpdf(yt))
    # Again check that the objective is finite and of the right data type.
    assert np.isfinite(B.to_numpy(objective))
    assert B.dtype(objective) == nps.dtype


def test_transform_positive(nps):
    model = nps.construct_convgnp(
        dim_x=1,
        dim_y=1,
        points_per_unit=16,
        unet_channels=(8, 16),
        transform="positive",
    )
    xc, yc, xt, yt = generate_data(nps, dim_x=1, dim_y=1)
    # Make data positive.
    yc = B.exp(yc)
    yt = B.exp(yt)

    pred = model(xc, yc, xt)
    objective = B.sum(pred.logpdf(yt))
    # Again check that the objective is finite and of the right data type.
    assert np.isfinite(B.to_numpy(objective))
    # Check that predictions and samples satisfy the constraint.
    assert B.all(pred.mean > 0)
    assert B.all(pred.sample() > 0)


def test_transform_bounded(nps):
    model = nps.construct_convgnp(
        dim_x=1,
        dim_y=1,
        points_per_unit=16,
        unet_channels=(8, 16),
        transform=(10, 11),
    )
    xc, yc, xt, yt = generate_data(nps, dim_x=1, dim_y=1)
    # Force data in the range `(10, 11)`.
    yc = 10 + 1 / (1 + B.exp(yc))
    yt = 10 + 1 / (1 + B.exp(yt))

    pred = model(xc, yc, xt)
    objective = B.sum(pred.logpdf(yt))
    # Again check that the objective is finite and of the right data type.
    assert np.isfinite(B.to_numpy(objective))
    # Check that predictions and samples satisfy the constraint.
    assert B.all(pred.mean > 10) and B.all(pred.mean < 11)
    assert B.all(pred.sample() > 10) and B.all(pred.sample() < 11)


def test_convgnp_auxiliary_variable(nps):
    model = nps.construct_convgnp(
        dim_x=2,
        dim_yc=(3, 1, 2),
        dim_aux_t=4,
        dim_yt=3,
        num_basis_functions=16,
        points_per_unit=16,
        likelihood="lowrank",
    )

    observed_data = (
        B.randn(nps.dtype, 16, 2, 10),
        B.randn(nps.dtype, 16, 3, 10),
    )
    aux_var1 = (
        B.randn(nps.dtype, 16, 2, 12),
        B.randn(nps.dtype, 16, 1, 12),
    )
    aux_var2 = (
        (B.randn(nps.dtype, 16, 1, 25), B.randn(nps.dtype, 16, 1, 35)),
        B.randn(nps.dtype, 16, 2, 25, 35),
    )
    aux_var_t = B.randn(nps.dtype, 16, 4, 15)
    pred = model(
        [observed_data, aux_var1, aux_var2],
        B.randn(nps.dtype, 16, 2, 15),
        aux_t=aux_var_t,
    )
    mean, var = pred.mean, pred.var

    # Check that the logpdf at the mean is finite and of the right data type.
    objective = B.sum(pred.logpdf(pred.mean))
    assert np.isfinite(B.to_numpy(objective))
    assert B.dtype(objective) == nps.dtype


@pytest.mark.flaky(rerun=3)
def test_convgnp_masking(nps):
    model = nps.construct_convgnp(
        num_basis_functions=16,
        points_per_unit=16,
        conv_arch="dws",
        dws_receptive_field=0.5,
        dws_layers=1,
        dws_channels=1,
        # Dividing by the density channel makes the forward very sensitive to the
        # numerics.
        divide_by_density=False,
    )
    xc, yc, xt, yt = generate_data(nps)

    # Predict without the final three points.
    pred = model(xc[:, :, :-3], yc[:, :, :-3], xt)
    # Predict using a mask instead.
    mask = B.to_numpy(B.ones(yc))  # Perform assignment in NumPy.
    mask[:, :, -3:] = 0
    mask = B.cast(B.dtype(yc), mask)
    pred_masked = model(xc, nps.Masked(yc, mask), xt)

    # Check that they coincide.
    approx(pred.mean, pred_masked.mean)
    approx(pred.var, pred_masked.var)

import lab as B
import numpy as np
import pytest
from itertools import product

from .util import nps as nps_fixed_dtype, approx, generate_data  # noqa


def generate_arch_variations(configs):
    varied_configs = []
    for config in configs:
        for variation in [
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
        ]:
            varied_configs.append(dict(config, **variation))
    return varied_configs


def product_kw_args(config, **kw_args):
    configs = []
    keys, values = zip(*kw_args.items())
    for variation in product(*values):
        configs.append(dict(config, **{k: v for (k, v) in zip(keys, variation)}))
    return configs


@pytest.fixture(
    params=[]
    # CNP:
    + product_kw_args(
        {
            "constructor": "construct_gnp",
            "num_basis_functions": 4,
            "dim_lv": 0,
        },
        dim_x=[1, 2],
        dim_y=[1, 2],
        likelihood=["het", "lowrank"],
    )
    # NP:
    + product_kw_args(
        {
            "constructor": "construct_gnp",
            "num_basis_functions": 4,
            "dim_lv": 3,
        },
        dim_x=[1, 2],
        dim_y=[1, 2],
        likelihood=["het", "lowrank"],
        lv_likelihood=["het", "dense"],
    )
    # ACNP:
    + product_kw_args(
        {
            "constructor": "construct_gnp",
            "num_basis_functions": 4,
            "dim_lv": 0,
        },
        dim_x=[1, 2],
        dim_y=[1, 2],
        likelihood=["het", "lowrank"],
    )
    # ANP:
    + product_kw_args(
        {
            "constructor": "construct_gnp",
            "num_basis_functions": 4,
            "dim_lv": 3,
        },
        dim_x=[1, 2],
        dim_y=[1, 2],
        likelihood=["het", "lowrank"],
        lv_likelihood=["het", "dense"],
    )
    # ConvCNP:
    + generate_arch_variations(
        product_kw_args(
            {
                "constructor": "construct_convgnp",
                "num_basis_functions": 4,
                "points_per_unit": 8,
                "dim_lv": 0,
            },
            dim_x=[1, 2],
            dim_y=[1, 2],
            likelihood=["het", "lowrank"],
        )
    )
    # ConvNP:
    + generate_arch_variations(
        product_kw_args(
            {
                "constructor": "construct_convgnp",
                "num_basis_functions": 4,
                "points_per_unit": 8,
                "dim_lv": 3,
            },
            dim_x=[1, 2],
            dim_y=[1, 2],
            likelihood=["het", "lowrank"],
            lv_likelihood=["het", "lowrank"],
        )
    )
    # FullConvGNP:
    + generate_arch_variations(
        [
            {
                "constructor": "construct_fullconvgnp",
                "points_per_unit": 8,
                "dim_x": 1,
                "dim_y": 1,
            }
        ]
    ),
    scope="module",
)
def config(request):
    return request.param


@pytest.fixture(params=[False, True], scope="module")
def nps(request, nps_fixed_dtype):
    nps = nps_fixed_dtype

    # Safely make a copy of `nps` so that we can modify the value of `dtype` without
    # the changes having side effects.

    class Namespace:
        pass

    nps_copy = Namespace()
    for attr in nps.__dir__():
        setattr(nps_copy, attr, getattr(nps, attr))
    nps = nps_copy

    # Use `float64`s or not?
    if request.param:
        nps.dtype = nps.dtype64

    return nps


@pytest.fixture(scope="module")
def model_sample(request, nps, config):
    # Construct model.
    nps.config = config  # Save the config for easier debugging.
    config = dict(config)
    constructor = getattr(nps, config["constructor"])
    del config["constructor"]
    model = constructor(**config, dtype=nps.dtype)

    # Run the model once to make sure all parameters exist.
    xc, yc, xt, yt = generate_data(nps, dim_x=config["dim_x"], dim_y=config["dim_y"])
    model(xc, yc, xt)

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

    def sample():
        return generate_data(nps, dim_x=config["dim_x"], dim_y=config["dim_y"])

    return model, sample


def check_prediction(nps, pred, yt):
    # Check that the log-pdf at the target data is finite and of the right data type.
    objective = B.sum(pred.logpdf(yt))
    assert np.isfinite(B.to_numpy(objective))
    assert B.dtype(objective) == nps.dtype

    # Check mean, variance, and samples.
    assert B.shape(pred.mean) == B.shape(yt)
    assert B.shape(pred.var) == B.shape(yt)
    assert B.shape(pred.sample()) == B.shape(yt)
    assert B.shape(pred.sample(2)) == (2,) + B.shape(yt)


@pytest.mark.flaky(reruns=3)
def test_forward(nps, model_sample):
    model, sample = model_sample
    xc, yc, xt, yt = sample()

    # Check passing in a non-empty context set.
    pred = model(xc, yc, xt, batch_size=2, unused_arg=None)
    check_prediction(nps, pred, yt)


@pytest.mark.flaky(reruns=3)
def test_elbo(nps, model_sample):
    model, sample = model_sample
    xc, yc, xt, yt = sample()
    elbo = B.mean(nps.elbo(model, xc, yc, xt, yt, num_samples=2))
    assert np.isfinite(B.to_numpy(elbo))


@pytest.mark.flaky(reruns=3)
def test_loglik(nps, model_sample):
    model, sample = model_sample
    xc, yc, xt, yt = sample()
    logpdfs = B.mean(nps.loglik(model, xc, yc, xt, yt, num_samples=2))
    assert np.isfinite(B.to_numpy(logpdfs))


@pytest.mark.flaky(reruns=3)
def test_predict(nps, model_sample):
    model, sample = model_sample
    xc, yc, xt, yt = sample()
    mean, var, samples = nps.predict(
        model,
        xc,
        yc,
        xt,
        num_samples=2,
        pred_num_samples=2,
    )
    assert B.shape(mean) == B.shape(yt)
    assert B.shape(var) == B.shape(yt)
    assert B.shape(samples) == (2,) + B.shape(yt)


@pytest.mark.flaky(reruns=3)
def test_batching(nps, model_sample):
    model, sample = model_sample
    xc, yc, xt, yt = sample()

    state = B.create_random_state(nps.dtype, seed=0)
    state2 = B.create_random_state(nps.dtype, seed=0)

    # Check that batching works correctly.
    _, pred = model(state, xc, yc, xt, batch_size=2, unused_arg=None)
    _, pred2 = model(
        state2,
        B.reshape(xc, 2, -1, *B.shape(xc, 1, 2)),
        B.reshape(yc, 2, -1, *B.shape(yc, 1, 2)),
        B.reshape(xt, 2, -1, *B.shape(xt, 1, 2)),
        batch_size=2,
        unused_arg=None,
    )
    approx(pred.mean, B.reshape(pred2.mean, -1, *B.shape(pred2.mean, 2, 3)))
    approx(pred.var, B.reshape(pred2.var, -1, *B.shape(pred2.var, 2, 3)))


@pytest.mark.flaky(reruns=3)
def test_empty_context(nps, model_sample):
    model, sample = model_sample
    xc, yc, xt, yt = sample()

    # Check passing the edge case of an empty context set.
    if isinstance(xc, B.Numeric):
        xc = xc[:, :, :0]
    elif isinstance(xc, tuple):
        xc = tuple(xci[:, :, :0] for xci in xc)
    else:
        raise RuntimeError("Failed to contruct empty context set.")
    yc = yc[:, :, :0]
    pred = model(xc, yc, xt, batch_size=2, unused_arg=None)
    check_prediction(nps, pred, yt)


@pytest.mark.flaky(reruns=3)
def test_recode(nps, model_sample):
    model, sample = model_sample
    xc, yc, xt, yt = sample()

    x_new = B.concat(
        xc,
        # Add new inputs which defy the extrema of the current context and target
        # inputs.
        B.max(xc, axis=-1, squeeze=False) + 1,
        B.max(xt, axis=-1, squeeze=False) + 1,
        axis=-1,
    )
    y_new = B.concat(yc, yc[:, :, -1:] + 1, yt[:, :, -1:] + 1, axis=-1)

    z, pz, h = nps.code_track(model.encoder, xc, yc, xt)
    z2, pz2, _ = nps.recode(model.encoder, x_new, y_new, h)

    approx(z, z2)

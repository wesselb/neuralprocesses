from itertools import product

import lab as B
import numpy as np
import pytest
from neuralprocesses.aggregate import Aggregate, AggregateInput
from neuralprocesses.parallel import Parallel
from plum import isinstance

from .util import approx, generate_data
from .util import nps as nps_fixed_dtype  # noqa


def generate_conv_arch_variations(configs):
    varied_configs = []
    for config in configs:
        for variation in [
            {
                # Standard UNet:
                "conv_arch": "unet",
                "unet_channels": (4, 8),
                "unet_kernels": (3, 5),
                "unet_strides": (2, 2),
                "unet_activations": (B.relu, B.tanh),
                "unet_resize_convs": False,
            },
            {
                # Fancy UNet:
                "conv_arch": "unet",
                "unet_channels": (4, 8),
                "unet_kernels": ((3, 3), 5),
                "unet_strides": (1, 2),
                "unet_activations": (B.relu, B.tanh),
                "unet_resize_convs": True,
            },
            {
                "conv_arch": "conv",
                "conv_channels": 8,
                "conv_layers": 2,
                "conv_receptive_field": 2,
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
    # CNP and GNP:
    + product_kw_args(
        {
            "constructor": "construct_gnp",
            "dim_embedding": 4,
            "width": 4,
            "num_basis_functions": 4,
            "dim_lv": 0,
        },
        dim_x=[1, 2],
        dim_y=[1, 2],
        likelihood=["het", "lowrank", "spikes-beta"],
    )
    # NP:
    + product_kw_args(
        {
            "constructor": "construct_gnp",
            "dim_embedding": 4,
            "width": 4,
            "num_basis_functions": 4,
            "dim_lv": 3,
        },
        dim_x=[1, 2],
        dim_y=[1, 2],
        likelihood=["het", "lowrank", "spikes-beta"],
        lv_likelihood=["het", "dense"],
    )
    # ACNP:
    + product_kw_args(
        {
            "constructor": "construct_agnp",
            "dim_embedding": 4,
            "num_heads": 2,
            "width": 4,
            "num_basis_functions": 4,
            "dim_lv": 0,
        },
        dim_x=[1, 2],
        dim_y=[1, 2],
        likelihood=["het", "lowrank", "spikes-beta"],
    )
    # ANP:
    + product_kw_args(
        {
            "constructor": "construct_agnp",
            "dim_embedding": 4,
            "num_heads": 2,
            "width": 4,
            "num_basis_functions": 4,
            "dim_lv": 3,
        },
        dim_x=[1, 2],
        dim_y=[1, 2],
        likelihood=["het", "lowrank", "spikes-beta"],
        lv_likelihood=["het", "dense"],
    )
    # ConvCNP and ConvGNP:
    + generate_conv_arch_variations(
        product_kw_args(
            {
                "constructor": "construct_convgnp",
                "num_basis_functions": 4,
                "points_per_unit": 8,
                "dim_lv": 0,
            },
            dim_x=[1, 2],
            dim_y=[1, 2],
            likelihood=["het", "lowrank", "spikes-beta"],
            encoder_scales_learnable=[True, False],
            decoder_scale_learnable=[True, False],
        )
    )
    # ConvNP:
    + generate_conv_arch_variations(
        product_kw_args(
            {
                "constructor": "construct_convgnp",
                "num_basis_functions": 4,
                "points_per_unit": 8,
                "dim_lv": 3,
            },
            dim_x=[1, 2],
            dim_y=[1, 2],
            likelihood=["het", "lowrank", "spikes-beta"],
            lv_likelihood=["het", "lowrank"],
        )
    )
    # FullConvGNP:
    + generate_conv_arch_variations(
        product_kw_args(
            {
                "constructor": "construct_fullconvgnp",
                "points_per_unit": 8,
                "dim_x": 1,
            },
            dim_y=[1, 2],
        )
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

    # Create a constructor which resamples the parameters of the model. This will ensure
    # that flaky tests which are rerun don't get stuck at particularly bad model
    # initialisations.

    if isinstance(nps.dtype, B.TFDType):

        def construct_model():
            new_weights = []
            for p in model.get_weights():
                new_weights.append(p + 1e-2 * B.randn(p))
            model.set_weights(new_weights)
            return model

    elif isinstance(nps.dtype, B.TorchDType):

        def construct_model():
            for p in model.parameters():
                p.data = p.data + 1e-2 * B.randn(p)
            return model

    else:
        raise RuntimeError("I don't know how to resample the parameters of the model.")

    def sample():
        if "likelihood" in config:
            binary = config["likelihood"] == "spikes-beta"
        else:
            binary = False
        return generate_data(
            nps,
            dim_x=config["dim_x"],
            dim_y=config["dim_y"],
            binary=binary,
        )

    return construct_model, sample


def shape(x, extra=()):
    if isinstance(x, Aggregate):
        return tuple(extra + B.shape(xi) for xi in x)
    else:
        return extra + B.shape(x)


def check_prediction(nps, pred, yt):
    # Ensure that there is sufficient noise.
    try:
        pred._noise = pred._noise + 1e-2 * B.eye(pred._noise)
    except AttributeError:
        pass

    # Check that the log-pdf at the target data is finite and of the right data type.
    objective = pred.logpdf(yt)
    assert B.rank(objective) == 1
    assert np.isfinite(B.to_numpy(B.sum(objective)))
    assert B.dtype(objective) == nps.dtype

    # Check mean, variance, and samples.
    assert shape(pred.mean) == shape(yt)
    assert shape(pred.var) == shape(yt)
    assert shape(pred.sample()) == shape(yt)
    assert shape(pred.sample(1)) == shape(yt, (1,))
    assert shape(pred.sample(2)) == shape(yt, (2,))


def _aggregate_xt_yt(xt, yt):
    xt = AggregateInput(*((xt, i) for i in range(B.shape(yt, 1))))
    yt = Aggregate(*(yt[:, i : i + 1, :] for i in range(B.shape(yt, 1))))
    return xt, yt


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.flaky(reruns=3)
def test_forward(nps, model_sample, aggregate):
    model, sample = model_sample
    model = model()
    xc, yc, xt, yt = sample()
    if aggregate:
        xt, yt = _aggregate_xt_yt(xt, yt)
    pred = model(xc, yc, xt, batch_size=2, unused_arg=None)
    check_prediction(nps, pred, yt)


@pytest.mark.parametrize("normalise", [False, True])
@pytest.mark.flaky(reruns=3)
def test_elbo(nps, model_sample, normalise):
    """Test the ELBO.

    We don't test aggregates here, because `nps.elbo` assumes a particular structure of
    the context sets in that case.
    """
    model, sample = model_sample
    model = model()
    xc, yc, xt, yt = sample()
    elbos = nps.elbo(model, xc, yc, xt, yt, num_samples=2, normalise=normalise)
    assert B.rank(elbos) == 1
    assert np.isfinite(B.to_numpy(B.sum(elbos)))
    assert B.dtype(elbos) == nps.dtype64


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("normalise", [False, True])
@pytest.mark.flaky(reruns=3)
def test_loglik(nps, model_sample, normalise, aggregate):
    model, sample = model_sample
    model = model()
    xc, yc, xt, yt = sample()
    if aggregate:
        xt, yt = _aggregate_xt_yt(xt, yt)
    logpdfs = nps.loglik(model, xc, yc, xt, yt, num_samples=2, normalise=normalise)
    assert B.rank(logpdfs) == 1
    assert np.isfinite(B.to_numpy(B.sum(logpdfs)))
    assert B.dtype(logpdfs) == nps.dtype64


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.flaky(reruns=3)
def test_predict(nps, model_sample, aggregate):
    model, sample = model_sample
    model = model()
    xc, yc, xt, yt = sample()
    if aggregate:
        xt, yt = _aggregate_xt_yt(xt, yt)
    mean, var, samples, noisy_samples = nps.predict(
        model,
        xc,
        yc,
        xt,
        num_samples=3,
        batch_size=2,
    )
    assert shape(mean) == shape(yt)
    assert shape(var) == shape(yt)
    assert shape(samples) == shape(yt, (3,))
    assert shape(noisy_samples) == shape(yt, (3,))


@pytest.mark.flaky(reruns=3)
def test_batching(nps, model_sample):
    model, sample = model_sample
    model = model()
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
    model = model()
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
    model = model()
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

    z, pz, h = nps.code_track(model.encoder, xc, yc, xt, root=True)
    z2, pz2, _ = nps.recode(model.encoder, x_new, y_new, h, root=True)

    approx(z, z2)


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_y", [1, 2])
@pytest.mark.parametrize(
    "constructor, config",
    [
        ("construct_gnp", {}),
        ("construct_agnp", {}),
        ("construct_convgnp", {"points_per_unit": 4, "unet_channels": (2,)}),
        ("construct_fullconvgnp", {"points_per_unit": 4, "unet_channels": (2,)}),
    ],
)
@pytest.mark.parametrize("dim_lv", [0, 2])
def test_data_eq(nps, dim_x, dim_y, constructor, config, dim_lv):
    if constructor == "construct_fullconvgnp" and (dim_x > 1 or dim_lv > 1):
        pytest.skip()
    gen = nps.construct_predefined_gens(nps.dtype, dim_x=dim_x, dim_y=dim_y)["eq"]
    model = getattr(nps, constructor)(
        dim_x=dim_x,
        dim_yc=(1,) * dim_y,
        dim_yt=dim_y,
        dim_lv=dim_lv,
        dtype=nps.dtype,
        **config,
    )
    batch = gen.generate_batch()
    pred = model(batch["contexts"], batch["xt"])
    check_prediction(nps, pred, batch["yt"])


@pytest.mark.parametrize(
    "constructor_kw_args",
    [
        ("construct_gnp", {}),
        ("construct_agnp", {}),
        ("construct_convgnp", {"points_per_unit": 4}),
        ("construct_fullconvgnp", {"points_per_unit": 4}),
    ],
)
def test_context_verification(nps, constructor_kw_args):
    constructor, kw_args = constructor_kw_args

    c = (B.randn(nps.dtype, 4, 1, 5), B.randn(nps.dtype, 4, 1, 5))
    xt = B.randn(nps.dtype, 4, 1, 15)

    # Test one context set.
    model = getattr(nps, constructor)(dim_yc=1, dtype=nps.dtype, **kw_args)
    pred1 = model(*c, xt)
    pred2 = model([c], xt)
    approx(pred1.mean, pred2.mean)
    with pytest.raises(AssertionError, match="(?i)got inputs and outputs in parallel"):
        model([c, c], xt)
    with pytest.raises(AssertionError, match="(?i) got inputs in parallel"):
        model(Parallel(c[0], c[0]), c[1], xt)
    with pytest.raises(AssertionError, match="(?i) got outputs in parallel"):
        model(c[0], Parallel(c[1], c[1]), xt)

    # Test two context sets.
    model = getattr(nps, constructor)(dim_yc=(1, 1), dtype=nps.dtype, **kw_args)
    model([c, c], xt)
    with pytest.raises(AssertionError, match="(?i)expected a parallel of elements"):
        model([c], xt)
    with pytest.raises(
        AssertionError,
        match="(?i)expected a parallel of 2 elements, but got 4 inputs and 4 outputs",
    ):
        model([c, c, c, c], xt)

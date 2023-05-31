import lab as B
import numpy as np
import pytest
from plum import Dispatcher

from neuralprocesses.augment import AugmentedInput
from neuralprocesses.mask import Masked
from .util import nps, remote_xfail, remote_skip  # noqa

_dispatch = Dispatcher()


@pytest.fixture(params=[1, 2], scope="module")
def dim_x(request):
    return request.param


@pytest.fixture(params=[1, 2], scope="module")
def dim_y(request):
    return request.param


@pytest.fixture(
    params=["eq", "matern", "weakly-periodic", "sawtooth", "mixture"],
    scope="module",
)
def predefined_gen(request, nps, dim_x, dim_y):
    gens = nps.construct_predefined_gens(
        nps.dtype,
        x_range_context=(0, 1),
        x_range_target=(1, 2),
        dim_x=dim_x,
        dim_y=dim_y,
        batch_size=4,
    )
    return gens[request.param]


def test_predefined_gens(nps, predefined_gen, dim_x, dim_y):
    # Limit the epoch to 10 batches.
    for _, batch in zip(range(10), predefined_gen.epoch()):
        # Unpack batch.
        cs = batch["contexts"]
        xt = batch["xt"]
        yt = batch["yt"]

        # Check the contexts.
        assert len(cs) == dim_y
        for xc, yc in cs:
            # Check the inputs.
            assert B.shape(xc, 0, 1) == (4, dim_x)
            assert B.shape(yc, 0, 1) == (4, 1)
            assert B.all(0 <= xc) and B.all(xc <= 1)
            assert B.dtype(xc) == nps.dtype

            # Check the outputs.
            assert B.shape(xc, 2) == B.shape(yc, 2)
            assert B.dtype(yc) == nps.dtype

        if dim_y == 1:
            # Check the target inputs.
            nt = B.shape(xt, 2)
            assert B.shape(xt, 0, 1) == (4, dim_x)
            assert B.all(1 <= xt) and B.all(xt <= 2)
            assert B.dtype(xt) == nps.dtype

            # Check the target outputs.
            assert B.shape(yt) == (4, 1, nt)
            assert B.dtype(yt) == nps.dtype
        else:
            # Check the target inputs.
            assert isinstance(xt, nps.AggregateInput)
            assert len(xt) == dim_y
            nts = []
            for i_expected, (xti, i) in enumerate(xt):
                assert i == i_expected
                assert B.shape(xti, 0, 1) == (4, dim_x)
                assert B.all(1 <= xti) and B.all(xti <= 2)
                assert B.dtype(xti) == nps.dtype
                nts.append(B.shape(xti, 2))

            # Check the target outputs.
            assert isinstance(yt, nps.Aggregate)
            assert len(yt) == dim_y
            for nti, yti in zip(nts, yt):
                assert B.shape(yti) == (4, 1, nti)
                assert B.dtype(yti) == nps.dtype


def check_batch_simple(nps, batch):
    # Check context sets.
    for xc, yc in batch["contexts"]:
        assert B.shape(xc, -1) > 0
        assert B.dtype(xc) == nps.dtype
        assert B.shape(yc, -1) > 0
        assert B.dtype(yc) == nps.dtype

    # Check target inputs.
    for xti, i in batch["xt"]:
        assert B.dtype(xti) == nps.dtype

    # Check target outputs.
    for yti in batch["yt"]:
        assert B.dtype(yti) == nps.dtype


@pytest.mark.parametrize(
    "mode", ["interpolation", "forecasting", "reconstruction", "random"]
)
@pytest.mark.parametrize("generator", ["PredPreyGenerator", "PredPreyRealGenerator"])
def test_predprey(nps, generator, mode):
    g = getattr(nps, generator)(nps.dtype, mode=mode)
    check_batch_simple(nps, g.generate_batch())


@remote_skip
@pytest.mark.parametrize(
    "mode", ["interpolation", "forecasting", "reconstruction", "random"]
)
@pytest.mark.parametrize("subset", ["train", "cv", "eval"])
def test_eeg(nps, mode, subset):
    g = nps.EEGGenerator(nps.dtype, mode=mode, subset=subset)
    check_batch_simple(nps, g.generate_batch())


@_dispatch
def _centred_around_europe(x: B.Numeric):
    return _lons_lats_centred_around_europe(x[:, 0, :], x[:, 1, :])


@_dispatch
def _centred_around_europe(x: tuple):
    return _lons_lats_centred_around_europe(x[0], x[1])


def _lons_lats_centred_around_europe(lons, lats):
    return (-20 <= B.mean(lons) <= 40) and (40 <= B.mean(lats) <= 75)


@_dispatch
def _bcn_form(x: B.Numeric):
    return B.rank(x) >= 3 and B.shape(x, 0) == 16


@_dispatch
def _bcn_form(x: tuple):
    return all(_bcn_form(xi) for xi in x)


@remote_xfail
@pytest.mark.parametrize("data_task", ["germany", "europe", "value"])
@pytest.mark.parametrize("context_sample", [False, True])
@pytest.mark.parametrize("target_square", [0, 10])
@pytest.mark.parametrize("target_elev", [False, True])
def test_temperature(
    nps,
    data_task,
    context_sample,
    target_square,
    target_elev,
):
    gen = nps.TemperatureGenerator(
        nps.dtype,
        batch_size=16,
        context_sample=context_sample,
        context_sample_factor=10,
        target_min=3,
        target_square=target_square,
        target_elev=target_elev,
        subset="train",
        data_task=data_task,
        data_fold=1,
    )
    batch = gen.generate_batch()

    # Check the contexts.
    xc, yc = batch["contexts"][0]  # Stations
    assert _bcn_form(xc)
    if context_sample:
        assert isinstance(yc, Masked)  # Context stations are masked to deal with NaNs.
        yc = yc.y
        assert B.shape(xc, -1) > 0
        assert B.shape(yc, -1) > 0
        assert _centred_around_europe(xc)
    else:
        assert B.shape(xc, -1) == 0
        assert B.shape(yc, -1) == 0
    assert _bcn_form(yc)

    xc, yc = batch["contexts"][1]  # Gridded data
    assert _bcn_form(xc)
    assert _bcn_form(yc)
    assert _centred_around_europe(xc)

    xc, yc = batch["contexts"][2]  # Gridded elevation
    assert isinstance(yc, Masked)  # Gridded elevation is masked to deal with NaNs.
    yc = yc.y
    assert _bcn_form(xc)
    assert _bcn_form(yc)
    assert _centred_around_europe(xc)

    xc, yc = batch["contexts"][3]  # Station elevation
    assert _bcn_form(xc)
    assert _bcn_form(yc)
    assert _centred_around_europe(xc)
    # Station elevations must be like the gridded elevation. E.g., this check that
    # they are normalised similarly.
    assert B.abs(B.mean(yc) - B.mean(batch["contexts"][2][1].y)) < 2

    # Check the targets.
    xt, yt = batch["xt"], batch["yt"]
    # The target inputs might be augmented.
    if target_elev:
        assert isinstance(xt, AugmentedInput)
        assert _bcn_form(xt.x)
        assert _bcn_form(xt.augmentation)
        assert B.shape(xt.x, -1) >= 3
        assert B.shape(xt.x, -1) == B.shape(xt.augmentation, -1)
        assert _centred_around_europe(xt.x)
    else:
        assert not isinstance(xt, AugmentedInput)
        assert _bcn_form(xt)
        assert B.shape(xt, -1) >= 3
        assert _centred_around_europe(xt)
    assert _bcn_form(yt)
    assert B.shape(yt, -1) >= 3

    # Check that all targets lie in the same square.
    if target_square > 0:
        if target_elev:
            assert isinstance(xt, AugmentedInput)
            xt = xt.x
        assert B.max(B.pw_dists(B.transpose(xt))) <= B.sqrt(2) * target_square


@remote_xfail
@pytest.mark.parametrize("data_task", ["germany", "europe", "value"])
def test_temperature_splitting(nps, data_task):
    # Check that the subsets and folds correctly divide things up.
    eval_masks = []

    for fold in [1, 2, 3, 4, 5]:
        masks = []
        for subset in ["train", "cv", "eval"]:
            masks.append(
                nps.TemperatureGenerator(
                    nps.dtype,
                    subset=subset,
                    data_task=data_task,
                    data_fold=fold,
                )._mask
            )
            if subset == "eval":
                eval_masks.append(masks[-1])

        # Check current subsets.
        assert (masks[0] | masks[1] | masks[2]).all()
        assert ~(masks[0] & masks[1]).any()
        assert ~(masks[1] & masks[2]).any()
        assert ~(masks[0] & masks[2]).any()

    # Check folds.
    assert np.logical_or.reduce(eval_masks).all()
    assert ~np.logical_and.reduce(eval_masks).any()

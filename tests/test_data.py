import lab as B
import pytest
from plum import Dispatcher
from neuralprocesses.augment import AugmentedInput
from neuralprocesses.mask import Masked

from .util import nps  # noqa


_dispatch = Dispatcher()


@pytest.fixture(params=[1, 2], scope="module")
def dim_x(request):
    return request.param


@pytest.fixture(params=[1, 2], scope="module")
def dim_y(request):
    return request.param


@pytest.fixture(
    params=["eq", "matern", "weakly-periodic", "sawtooth", "mixture", "predprey"],
    scope="module",
)
def gen(request, nps, dim_x, dim_y):
    gens = nps.construct_predefined_gens(
        nps.dtype,
        x_range_context=(0, 1),
        x_range_target=(1, 2),
        dim_x=dim_x,
        dim_y=dim_y,
        batch_size=4,
    )
    if dim_x == 1 and dim_y == 2:
        gens["predprey"] = nps.PredPreyGenerator(
            nps.dtype,
            dist_x_context=nps.UniformContinuous(0, 1),
            dist_x_target=nps.UniformContinuous(1, 2),
            batch_size=4,
        )
    try:
        return gens[request.param]
    except KeyError:
        # TODO: Can we silently skip tests?
        pytest.skip()


def test_predefined_gens(nps, gen, dim_x, dim_y):
    # Limit the epoch to 10 batches.
    for _, batch in zip(range(10), gen.epoch()):

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

            # Check the outputs.
            assert B.shape(xc, 2) == B.shape(yc, 2)

        if dim_y == 1:
            # Check the target inputs.
            nt = B.shape(xt, 2)
            assert B.shape(xt, 0, 1) == (4, dim_x)
            assert B.all(1 <= xt) and B.all(xt <= 2)

            # Check the target outputs.
            assert B.shape(yt) == (4, 1, nt)
        else:
            # Check the target inputs.
            assert isinstance(xt, nps.AggregateInput)
            assert len(xt) == dim_y
            nts = []
            for i_expected, (xti, i) in enumerate(xt):
                assert i == i_expected
                assert B.shape(xti, 0, 1) == (4, dim_x)
                assert B.all(1 <= xti) and B.all(xti <= 2)
                nts.append(B.shape(xti, 2))

            # Check the target outputs.
            assert isinstance(yt, nps.Aggregate)
            assert len(yt) == dim_y
            for nti, yti in zip(nts, yt):
                assert B.shape(yti) == (4, 1, nti)


@_dispatch
def _within_germany(x: B.Numeric):
    return _lons_lats_within_germany(x[:, 0, :], x[:, 1, :])


@_dispatch
def _within_germany(x: tuple):
    return _lons_lats_within_germany(x[0], x[1])


def _lons_lats_within_germany(lons, lats):
    return (6 <= B.mean(lons) <= 16) and (47 <= B.mean(lats) <= 55)


@_dispatch
def _bcn_form(x: B.Numeric):
    return B.rank(x) >= 3 and B.shape(x, 0) == 16


@_dispatch
def _bcn_form(x: tuple):
    return all(_bcn_form(xi) for xi in x)


# @pytest.mark.xfail()
@pytest.mark.parametrize("subset", ["train", "cv", "eval"])
@pytest.mark.parametrize("context_sample", [False, True])
@pytest.mark.parametrize("target_square", [0, 2])
@pytest.mark.parametrize("target_elev", [False, True])
def test_temperature(
    nps,
    subset,
    context_sample,
    target_square,
    target_elev,
):
    gen = nps.TemperatureGenerator(
        nps.dtype,
        batch_size=16,
        context_sample=context_sample,
        target_min=15,
        target_square=target_square,
        target_elev=target_elev,
        subset=subset,
    )
    batch = gen.generate_batch()

    # Check the contexts.
    xc, yc = batch["contexts"][0]  # Stations
    assert isinstance(yc, Masked)  # Context stations are masked to deal with NaNs.
    yc = yc.y
    assert _bcn_form(xc)
    assert _bcn_form(yc)
    if context_sample:
        assert B.shape(xc, -1) > 0
        assert B.shape(yc, -1) > 0
        assert _within_germany(xc)
    else:
        assert B.shape(xc, -1) == 0
        assert B.shape(yc, -1) == 0

    xc, yc = batch["contexts"][1]  # Gridded data
    assert _bcn_form(xc)
    assert _bcn_form(yc)
    assert _within_germany(xc)

    xc, yc = batch["contexts"][2]  # Gridded elevation
    assert isinstance(yc, Masked)  # Gridded elevation is masked to deal with NaNs.
    yc = yc.y
    assert _bcn_form(xc)
    assert _bcn_form(yc)
    assert _within_germany(xc)

    # Check the targets.
    xt, yt = batch["xt"], batch["yt"]
    # The target inputs might be augmented.
    if target_elev:
        assert isinstance(xt, AugmentedInput)
        assert _bcn_form(xt.x)
        assert _bcn_form(xt.augmentation)
        assert B.shape(xt.x, -1) >= 15
        assert B.shape(xt.x, -1) == B.shape(xt.augmentation, -1)
        assert _within_germany(xt.x)
    else:
        assert not isinstance(xt, AugmentedInput)
        assert _bcn_form(xt)
        assert B.shape(xt, -1) >= 15
        assert _within_germany(xt)
    assert _bcn_form(yt)
    assert B.shape(yt, -1) >= 15

    # Check that all targets lie in the same square.
    if target_square > 0:
        if target_elev:
            assert isinstance(xt, AugmentedInput)
            xt = xt.x
        assert B.max(B.pw_dists(B.transpose(xt))) <= B.sqrt(2) * target_square

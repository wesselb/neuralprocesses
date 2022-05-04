import lab as B
import pytest

from .util import nps  # noqa


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
    for _ in range(10):
        batch = gen.generate_batch()

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

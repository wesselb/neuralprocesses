import lab as B

from .test_architectures import check_prediction, generate_data
from .util import nps  # noqa


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

    check_prediction(nps, pred, yt)
    # Check that predictions and samples satisfy the constraint.
    assert B.all(pred.mean > 0)
    assert B.all(pred.sample(2) > 0)


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
    check_prediction(nps, pred, yt)
    # Check that predictions and samples satisfy the constraint.
    assert B.all(pred.mean > 10) and B.all(pred.mean < 11)
    assert B.all(pred.sample() > 10) and B.all(pred.sample() < 11)

import lab as B
import numpy as np
import pytest

from .test_architectures import generate_data
from .util import approx, generate_data, nps  # noqa


@pytest.mark.parametrize("dim_lv", [0, 4])
def test_loglik_batching(nps, dim_lv):
    model = nps.construct_gnp(dim_lv=dim_lv)
    xc, yc, xt, yt = generate_data(nps)
    # Test a high number of samples, a number which also isn't a multiple of the batch
    # size.
    logpdfs = B.mean(
        nps.loglik(model, xc, yc, xt, yt, num_samples=4000, batch_size=128)
    )
    assert np.isfinite(B.to_numpy(logpdfs))


def test_ar_predict_without_aggregate(nps):
    xc, yc, xt, yt = generate_data(nps, dim_x=2, dim_y=3)
    convcnp = nps.construct_gnp(dim_x=2, dim_yc=(1, 1, 1), dim_yt=3, dtype=nps.dtype)

    # Perform AR by using `AggregateInput` manually.
    _, mean1, var1, ft1, yt1 = nps.ar_predict(
        B.create_random_state(nps.dtype, seed=0),
        convcnp,
        [(xc, yc[:, 0:1, :]), (xc, yc[:, 1:2, :]), (xc, yc[:, 2:3, :])],
        nps.AggregateInput((xt, 0), (xt, 1), (xt, 2)),
    )
    mean1 = B.concat(*mean1, axis=-2)
    var1 = B.concat(*var1, axis=-2)
    ft1 = B.concat(*ft1, axis=-2)
    yt1 = B.concat(*yt1, axis=-2)

    # Let the package work its magic.
    _, mean2, var2, ft2, yt2 = nps.ar_predict(
        B.create_random_state(nps.dtype, seed=0),
        convcnp,
        xc,
        yc,
        xt,
    )

    # Check that the two give identical results.
    approx(mean1, mean2)
    approx(var1, var2)
    approx(ft1, ft2)
    approx(yt1, yt2)

import lab as B
import numpy as np
import pytest

from .util import nps, generate_data  # noqa


@pytest.mark.parametrize("float64", [False, True])
@pytest.mark.parametrize(
    "construct_name, kw_args",
    [
        (model, {"dim_x": dim_x, "dim_y": dim_y, "likelihood": lik})
        for model in ["construct_gnp", "construct_convgnp"]
        for dim_x in [1, 2]
        for dim_y in [1, 2]
        for lik in ["het", "lowrank", "lowrank-correlated"]
    ],
)
def test_architectures(nps, float64, construct_name, kw_args):
    if float64:
        nps.dtype = nps.dtype64
    model = getattr(nps, construct_name)(**kw_args, dtype=nps.dtype)
    xc, yc, xt, yt = generate_data(nps, dim_x=kw_args["dim_x"], dim_y=kw_args["dim_y"])
    pred = model(xc, yc, xt)
    objective = B.sum(pred.logpdf(yt))
    # Check that the objective is finite and of the right data type.
    assert np.isfinite(B.to_numpy(objective))
    assert B.dtype(objective) == nps.dtype

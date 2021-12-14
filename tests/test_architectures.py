import lab as B
import numpy as np
import pytest

from .util import nps, generate_data  # noqa


@pytest.mark.parametrize("float64", [False, True])
@pytest.mark.parametrize(
    "construct_name, kw_args",
    [
        ("construct_cnp", {"dim_x": 1, "dim_y": 2}),
        ("construct_cnp", {"dim_x": 2, "dim_y": 2}),
        ("construct_convgnp", {"dim_x": 1, "dim_y": 2, "likelihood": "het"}),
        ("construct_convgnp", {"dim_x": 2, "dim_y": 2, "likelihood": "het"}),
        ("construct_convgnp", {"dim_x": 1, "dim_y": 2, "likelihood": "lowrank"}),
        ("construct_convgnp", {"dim_x": 2, "dim_y": 2, "likelihood": "lowrank"}),
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

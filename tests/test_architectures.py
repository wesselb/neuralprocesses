import lab as B
import numpy as np
import pytest

from .util import nps, generate_data  # noqa


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
def test_architectures(nps, construct_name, kw_args):
    model = getattr(nps, construct_name)(**kw_args)
    xc, yc, xt, yt = generate_data(nps, dim_x=kw_args["dim_x"], dim_y=kw_args["dim_y"])
    pred = model(xc, yc, xt)
    assert np.isfinite(B.to_numpy(B.sum(pred.logpdf(yt))))

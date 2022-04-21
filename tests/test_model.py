import lab as B
import numpy as np
import pytest

from .test_architectures import generate_data
from .util import nps  # noqa


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

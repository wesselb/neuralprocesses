import numpy as np
import lab as B

from .test_architectures import generate_data
from .util import nps  # noqa


def test_loglik_batching(nps):
    model = nps.construct_gnp()
    xc, yc, xt, yt = generate_data(nps)
    # Test a high number of samples, a number which also isn't a multiple of the batch
    # size.
    logpdfs = B.mean(
        nps.loglik(model, xc, yc, xt, yt, num_samples=4000, batch_size=128)
    )
    assert np.isfinite(B.to_numpy(logpdfs))

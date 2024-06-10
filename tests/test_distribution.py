import lab as B
import scipy.stats as stats

import torch
from neuralprocesses.dist.beta import Beta
from neuralprocesses.dist.gamma import Gamma

from .test_architectures import check_prediction, generate_data
from .util import approx, nps  # noqa


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


def test_beta_correctness():
    """Test the correctness of the beta distribution."""
    beta = Beta(B.cast(torch.float64, 0.2), B.cast(torch.float64, 0.8), 0)
    beta_ref = stats.beta(0.2, 0.8)

    sample = beta.sample()
    approx(beta.logpdf(sample), beta_ref.logpdf(sample))
    approx(beta.mean, beta_ref.mean())
    approx(beta.var, beta_ref.var())

    # Test dimensionality argument.
    for d in range(4):
        beta = Beta(beta.alpha, beta.beta, d)
        assert beta.logpdf(beta.sample(1, 2, 3)).shape == (1, 2, 3)[: 3 - d]


def test_gamma():
    """Test the correctness of the gamma distribution."""
    gamma = Gamma(B.cast(torch.float64, 2), B.cast(torch.float64, 0.8), 0)
    gamma_ref = stats.gamma(2, scale=0.8)

    sample = gamma.sample()
    approx(gamma.logpdf(sample), gamma_ref.logpdf(sample))
    approx(gamma.mean, gamma_ref.mean())
    approx(gamma.var, gamma_ref.var())

    # Test dimensionality argument.
    for d in range(4):
        gamma = Gamma(gamma.k, gamma.scale, d)
        assert gamma.logpdf(gamma.sample(1, 2, 3)).shape == (1, 2, 3)[: 3 - d]

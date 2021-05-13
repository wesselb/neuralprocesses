import lab.torch as B
import pytest
import torch

import neuralprocesses.gnp as gnp

# noinspection PyUnresolvedReferences
from .util import context_set, target_set


def test_gnp(context_set, target_set):
    model = gnp.GNP(y_target_dim=B.shape(target_set[1])[2])
    mean, cov = model(*context_set, target_set[0])
    pred = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)

    # Check logpdf computation. This only works for one-dimensional outputs.
    assert B.shape(target_set[1])[2] == 1
    logpdf = pred.log_prob(target_set[1][:, :, 0])

    # Check output.
    assert B.all(torch.isfinite(logpdf))
    assert B.shape(logpdf) == B.shape(target_set[1])[:1]


def test_gnp_x_target_check(context_set, target_set):
    # Must provide target inputs:
    model = gnp.GNP()
    model(*context_set, target_set[0])
    with pytest.raises(ValueError):
        model(*context_set)

    # May provide target inputs:
    model = gnp.GNP(x_target=0.5 * target_set[0])
    diff = model(*context_set)[0] - model(*context_set, target_set[0])[0]
    assert B.sum(B.abs(diff)) > 1e-2

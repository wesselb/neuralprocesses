import lab as B
import pytest
from ..util import nps, approx  # noqa
from neuralprocesses.dist import MultiOutputNormal


_missing_dists = []

# Dense:
mean = B.randn(4, 3, 10)
var = B.randn(4, 3, 10, 3, 10)
# Make variance positive definite.
var = B.reshape(var, 4, 30, 30)
var = var @ B.transpose(var)
var = B.reshape(var, 4, 3, 10, 3, 10)
noise = B.rand(4, 3, 10)
_missing_dists.append(
    (
        MultiOutputNormal.dense(
            B.reshape(mean, 4, 30),
            B.reshape(var, 4, 30, 30),
            B.reshape(noise, 4, 30),
            (3, 10),
        ),
        MultiOutputNormal.dense(
            B.reshape(mean[:, :2, :], 4, 20),
            B.reshape(var[:, :2, :, :2, :], 4, 20, 20),
            B.reshape(noise[:, :2, :], 4, 20),
            (2, 10),
        ),
    )
)

# Diagonal:
mean = B.randn(4, 3, 10)
noise = B.rand(4, 3, 10)
_missing_dists.append(
    (
        MultiOutputNormal.diagonal(
            B.reshape(mean, 4, 30),
            B.reshape(noise, 4, 30),
            (3, 10),
        ),
        MultiOutputNormal.diagonal(
            B.reshape(mean[:, :2, :], 4, 20),
            B.reshape(noise[:, :2, :], 4, 20),
            (2, 10),
        ),
    )
)

# Low rank:
mean = B.randn(4, 3, 10)
var_factor = B.rand(4, 3, 10, 7)
noise = B.rand(4, 3, 10)
_missing_dists.append(
    (
        MultiOutputNormal.lowrank(
            B.reshape(mean, 4, 30),
            B.reshape(var_factor, 4, 30, 7),
            B.reshape(noise, 4, 30),
            (3, 10),
        ),
        MultiOutputNormal.lowrank(
            B.reshape(mean[:, :2, :], 4, 20),
            B.reshape(var_factor[:, :2, :, :], 4, 20, 7),
            B.reshape(noise[:, :2, :], 4, 20),
            (2, 10),
        ),
    )
)


@pytest.mark.parametrize("d, d_ref", _missing_dists)
def test_monormal_missing(nps, d, d_ref):
    y_ref = B.randn(4, 3, 10)
    y = y_ref.copy()
    y[:, 2, :] = B.nan
    approx(d.logpdf(y), d_ref.logpdf(y_ref[:, :2, :]))

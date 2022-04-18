import lab as B
from stheno import Normal

from . import _dispatch
from .dist import MultiOutputNormal
from .util import register_module

__all__ = ["HeterogeneousGaussianLikelihood", "LowRankGaussianLikelihood"]


@register_module
class HeterogeneousGaussianLikelihood:
    """Gaussian likelihood with heterogeneous noise.

    Args:
        epsilon (float, optional): Smallest allowable variance. Defaults to `1e-6`.

    Args:
        epsilon (float): Smallest allowable variance.
    """

    @_dispatch
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "HeterogeneousGaussianLikelihood()"


@_dispatch
def code(
    coder: HeterogeneousGaussianLikelihood,
    xz,
    z: B.Numeric,
    x,
    *,
    dtype_lik=None,
    noiseless=False,
    **kw_args,
):
    dim_y = B.shape(z, 1) // 2
    mean = z[:, :dim_y, :]
    noise = coder.epsilon + B.softplus(z[:, dim_y:, :])

    # Cast parameters to the right data type.
    if dtype_lik:
        mean = B.cast(dtype_lik, mean)
        noise = B.cast(dtype_lik, noise)

    # Make a noiseless prediction.
    if noiseless:
        with B.on_device(noise):
            noise = B.zeros(noise)

    return xz, MultiOutputNormal.diagonal(mean, noise)


@register_module
class LowRankGaussianLikelihood:
    """Gaussian likelihood with low-rank noise.

    Args:
        rank (int): Rank of the low-rank part of the noise variance.
        epsilon (float, optional): Smallest allowable variance. Defaults to `1e-6`.

    Attributes:
        rank (int): Rank of the low-rank part of the noise variance.
        epsilon (float): Smallest allowable diagonal variance.
    """

    @_dispatch
    def __init__(self, rank, epsilon: float = 1e-6):
        self.rank = rank
        self.epsilon = epsilon

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"LowRankGaussianLikelihood(rank={self.rank})"


@_dispatch
def code(
    coder: LowRankGaussianLikelihood,
    xz,
    z: B.Numeric,
    x,
    *,
    noiseless=False,
    dtype_lik=None,
    **kw_args,
):
    dim_y = B.shape(z, 1) // (2 + coder.rank)
    dim_inner = B.shape(z, 1) - 2 * dim_y
    mean = z[:, :dim_y, :]
    noise = coder.epsilon + B.softplus(z[:, dim_y : 2 * dim_y, :])
    var_factor = z[:, 2 * dim_y :, :] / B.sqrt(dim_inner)

    # Cast the parameters before constructing the distribution.
    if dtype_lik:
        mean = B.cast(dtype_lik, mean)
        noise = B.cast(dtype_lik, noise)
        var_factor = B.cast(dtype_lik, var_factor)

    # Make a noiseless prediction.
    if noiseless:
        with B.on_device(noise):
            noise = B.zero(noise)

    pred = MultiOutputNormal.lowrank(mean, noise, var_factor)

    # If the noise was removed, convert the variance to dense, because the matrix
    # inversion lemma will otherwise fail.
    if noiseless:
        pred = MultiOutputNormal(
            Normal(pred.normal.mean, B.dense(pred.normal.var)),
            pred.num_outputs,
        )

    return xz, pred

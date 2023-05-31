from functools import reduce
from operator import mul

import lab as B
from lab.util import resolve_axis

from . import _dispatch
from .aggregate import AggregateInput, Aggregate
from .datadims import data_dims
from .dist import MultiOutputNormal, Dirac, SpikesSlab, Beta
from .parallel import Parallel
from .util import register_module, split, split_dimension

__all__ = [
    "DeterministicLikelihood",
    "HeterogeneousGaussianLikelihood",
    "LowRankGaussianLikelihood",
    "DenseGaussianLikelihood",
]


def _vectorise(z, num, offset=0):
    """Vectorise the last few dimensions of a tensor.

    Args:
        z (tensor): Tensor to vectorise.
        num (int): Number of dimensions to vectorise.
        offset (int, optional): Don't vectorise the last few channels, but leave
            `offset` channels at the end untouched.

    Returns:
        tensor: Compressed version of `z`.
        shape: Shape of the compressed dimensions before compressing.
    """
    # Convert to positive indices for easier indexing.
    i1 = resolve_axis(z, -num - offset)
    i2 = resolve_axis(z, -(offset + 1)) + 1
    shape = B.shape(z)
    shape_before = shape[:i1]
    shape_compressed = shape[i1:i2]
    shape_after = shape[i2:]
    z = B.reshape(z, *shape_before, reduce(mul, shape_compressed, 1), *shape_after)
    return z, shape_compressed


class AbstractLikelihood:
    """A likelihood."""


@register_module
class DeterministicLikelihood(AbstractLikelihood):
    """Deterministic likelihood."""


@_dispatch
def code(
    coder: DeterministicLikelihood,
    xz,
    z,
    x,
    *,
    dtype_lik=None,
    **kw_args,
):
    d = data_dims(xz)

    if dtype_lik:
        z = B.cast(dtype_lik, z)

    return xz, Dirac(z, d)


@register_module
class HeterogeneousGaussianLikelihood(AbstractLikelihood):
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
        return f"HeterogeneousGaussianLikelihood(epsilon={self.epsilon!r})"


@_dispatch
def code(
    coder: HeterogeneousGaussianLikelihood,
    xz,
    z,
    x,
    *,
    dtype_lik=None,
    **kw_args,
):
    mean, noise, shape = _het(coder, xz, z)

    # Cast parameters to the right data type.
    if dtype_lik:
        mean = B.cast(dtype_lik, mean)
        noise = B.cast(dtype_lik, noise)
    return xz, MultiOutputNormal.diagonal(mean, noise, shape)


@_dispatch
def _het(
    coder: HeterogeneousGaussianLikelihood,
    xz: AggregateInput,
    z: Aggregate,
):
    means, noises, shapes = zip(*[_het(coder, xzi, zi) for (xzi, _), zi in zip(xz, z)])

    # Concatenate into one big Gaussian.
    mean = B.concat(*means, axis=-1)
    noise = B.concat(*noises, axis=-1)
    shape = Aggregate(*shapes)

    return mean, noise, shape


@_dispatch
def _het(coder: HeterogeneousGaussianLikelihood, xz, z: B.Numeric):
    d = data_dims(xz)
    dim_y = B.shape(z, -d - 1) // 2

    z_mean, z_noise = split(z, (dim_y, dim_y), -d - 1)

    # Vectorise the data. Record the shape!
    z_mean, shape = _vectorise(z_mean, d + 1)
    z_noise, _ = _vectorise(z_noise, d + 1)

    # Transform into parameters.
    mean = z_mean
    noise = coder.epsilon + B.softplus(z_noise)

    return mean, noise, shape


@register_module
class LowRankGaussianLikelihood(AbstractLikelihood):
    """Gaussian likelihood with low-rank covariance and heterogeneous noise.

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
        return (
            f"LowRankGaussianLikelihood(rank={self.rank!r}, epsilon={self.epsilon!r})"
        )


@_dispatch
def code(
    coder: LowRankGaussianLikelihood,
    xz,
    z,
    x,
    *,
    dtype_lik=None,
    **kw_args,
):
    mean, var_factor, noise, shape = _lowrank(coder, xz, z)

    # Cast the parameters before constructing the distribution.
    if dtype_lik:
        mean = B.cast(dtype_lik, mean)
        var_factor = B.cast(dtype_lik, var_factor)
        noise = B.cast(dtype_lik, noise)

    return xz, MultiOutputNormal.lowrank(mean, var_factor, noise, shape)


@_dispatch
def _lowrank(coder: LowRankGaussianLikelihood, xz: AggregateInput, z: Aggregate):
    means, var_factors, noises, shapes = zip(
        *[_lowrank(coder, xzi, zi) for (xzi, _), zi in zip(xz, z)]
    )

    # Concatenate into one big Gaussian.
    mean = B.concat(*means, axis=-1)
    var_factor = B.concat(*var_factors, axis=-2)
    noise = B.concat(*noises, axis=-1)
    shape = Aggregate(*shapes)

    return mean, var_factor, noise, shape


@_dispatch
def _lowrank(coder: LowRankGaussianLikelihood, xz, z: B.Numeric):
    d = data_dims(xz)

    dim_y = B.shape(z, -d - 1) // (2 + coder.rank)
    dim_inner = B.shape(z, -d - 1) - 2 * dim_y

    z_mean, z_var_factor, z_noise = split(z, (dim_y, coder.rank * dim_y, dim_y), -d - 1)
    # Split of the ranks of the factor.
    z_var_factor = split_dimension(z_var_factor, -d - 1, (coder.rank, dim_y))

    # Vectorise the data. Record the shape!
    z_mean, shape = _vectorise(z_mean, d + 1)
    z_var_factor, _ = _vectorise(z_var_factor, d + 1)
    z_noise, _ = _vectorise(z_noise, d + 1)

    # Put the dimensions of the factor last, because that it what the constructor
    # assumes.
    z_var_factor = B.transpose(z_var_factor)

    # Transform into parameters.
    mean = z_mean
    # Intuitively, roughly `var_factor ** 2` summed along the columns will determine the
    # output variances. We divide by the square root of the number of columns to
    # stabilise this.
    var_factor = z_var_factor / B.shape(z_var_factor, -1)
    noise = coder.epsilon + B.softplus(z_noise)

    return mean, var_factor, noise, shape


@register_module
class DenseGaussianLikelihood(AbstractLikelihood):
    """Gaussian likelihood with dense covariance matrix and heterogeneous noise.

    Args:
        epsilon (float, optional): Smallest allowable variance. Defaults to `1e-6`.

    Args:
        epsilon (float): Smallest allowable variance.
        transform_var (bool): Ensure that the covariance matrix is positive definite by
            multiplying with itself.
    """

    @_dispatch
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return r"DenseGaussianLikelihood(epsilon={self.epsilon!r})"


@_dispatch
def code(
    coder: DenseGaussianLikelihood,
    xz: Parallel,
    z: Parallel,
    x,
    *,
    dtype_lik=None,
    **kw_args,
):
    z_mean_noise, z_var = z

    mean, var, noise, shape = _dense(coder, *xz, *z)

    # Cast parameters to the right data type.
    if dtype_lik:
        mean = B.cast(dtype_lik, mean)
        var = B.cast(dtype_lik, var)
        noise = B.cast(dtype_lik, noise)

    # Return the inputs for the mean. The inputs for the variance will have been
    # duplicated.
    xz = xz[0]

    return xz, MultiOutputNormal.dense(mean, var, noise, shape)


@_dispatch
def _dense(
    coder: DenseGaussianLikelihood,
    xz_mean,
    xz_var,
    z_mean: B.Numeric,
    z_var: B.Numeric,
):
    mean, noise, shape = _dense_mean(coder, xz_mean, z_mean)
    var = _dense_var(coder, xz_var, z_var)
    return mean, var, noise, shape


@_dispatch
def _dense(
    coder: DenseGaussianLikelihood,
    xz_mean: AggregateInput,
    xz_var: AggregateInput,
    z_mean: Aggregate,
    z_var: Aggregate,
):
    means, noises, shapes = zip(
        *[_dense_mean(coder, xzi, zi) for (xzi, _), zi in zip(xz_mean, z_mean)]
    )
    vars = [
        [_dense_var(coder, xzij, zij) for (xzij, _), zij in zip(xzi, zi)]
        for (xzi, _), zi in zip(xz_var, z_var)
    ]

    # Concatenate everything into one big Gaussian.
    mean = B.concat(*means, axis=-1)
    var = B.concat2d(*vars)
    noise = B.concat(*noises, axis=-1)
    shape = Aggregate(*shapes)

    return mean, var, noise, shape


@_dispatch
def _dense_mean(coder: DenseGaussianLikelihood, xz, z: B.Numeric):
    d = data_dims(xz)
    dim_y = B.shape(z, -d - 1) // 2

    z_mean, z_noise = split(z, (dim_y, dim_y), -d - 1)

    # Vectorise the data. Record the shape!
    z_mean, shape = _vectorise(z_mean, d + 1)
    z_noise, _ = _vectorise(z_noise, d + 1)

    # Transform into parameters.
    mean = z_mean
    noise = coder.epsilon + B.softplus(z_noise)

    return mean, noise, shape


@_dispatch
def _dense_var(coder: DenseGaussianLikelihood, xz, z: B.Numeric):
    d = data_dims(xz) // 2

    # First vectorise inner channels and then vectorise outer channels.
    z, _ = _vectorise(z, d + 1, offset=d + 1)
    z, _ = _vectorise(z, d + 1)

    return z


@register_module
class SpikesBetaLikelihood(AbstractLikelihood):
    """Gaussian likelihood with heterogeneous noise.

    Args:
        epsilon (float, optional): Tolerance for equality checking. Defaults to `1e-6`.

    Args:
        epsilon (float): Tolerance for equality checking.
    """

    @_dispatch
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"SpikesBetaLikelihood(epsilon={self.epsilon!r})"


@_dispatch
def code(
    coder: SpikesBetaLikelihood,
    xz,
    z,
    x,
    *,
    dtype_lik=None,
    **kw_args,
):
    alpha, beta, logp0, logp1, logps, d = _spikesbeta(coder, xz, z)

    # Cast parameters to the right data type.
    if dtype_lik:
        alpha = B.cast(dtype_lik, alpha)
        beta = B.cast(dtype_lik, beta)
        logp0 = B.cast(dtype_lik, logp0)
        logp1 = B.cast(dtype_lik, logp1)
        logps = B.cast(dtype_lik, logps)

    # Create the spikes vector.
    with B.on_device(z):
        dtype = dtype_lik or B.dtype(z)
        spikes = B.stack(B.one(dtype), B.zero(dtype))

    return xz, SpikesSlab(
        spikes,
        Beta(alpha, beta, d),
        B.stack(logp0, logp1, logps, axis=-1),
        d,
        epsilon=coder.epsilon,
    )


@_dispatch
def _spikesbeta(
    coder: SpikesBetaLikelihood,
    xz: AggregateInput,
    z: Aggregate,
):
    alphas, betas, logp0s, logp1s, logpss, ds = zip(
        *[_spikesbeta(coder, xzi, zi) for (xzi, _), zi in zip(xz, z)]
    )

    # Concatenate into one big distribution.
    alpha = Aggregate(*alphas)
    beta = Aggregate(*betas)
    logp0 = Aggregate(*logp0s)
    logp1 = Aggregate(*logp1s)
    logps = Aggregate(*logpss)
    d = Aggregate(*ds)

    return alpha, beta, logp0, logp1, logps, d


@_dispatch
def _spikesbeta(coder: SpikesBetaLikelihood, xz, z: B.Numeric):
    d = data_dims(xz)
    dim_y = B.shape(z, -d - 1) // 5

    z_alpha, z_beta, z_logp0, z_logp1, z_logps = split(
        z, (dim_y, dim_y, dim_y, dim_y, dim_y), -d - 1
    )

    # Transform into parameters.
    alpha = 1e-3 + B.softplus(z_alpha)
    beta = 1e-3 + B.softplus(z_beta)
    logp0 = z_logp0
    logp1 = z_logp1
    logps = z_logps

    return alpha, beta, logp0, logp1, logps, d + 1

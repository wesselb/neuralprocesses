import lab as B
from lab.util import resolve_axis
from stheno import Normal

from . import _dispatch
from .dist import MultiOutputNormal, Dirac
from .parallel import Parallel
from .util import register_module, data_dims, split_channels

__all__ = [
    "DeterministicLikelihood",
    "HeterogeneousGaussianLikelihood",
    "LowRankGaussianLikelihood",
    "DenseGaussianLikelihood",
]


@register_module
class DeterministicLikelihood:
    """Deterministic likelihood."""


@_dispatch
def code(
    coder: DeterministicLikelihood,
    xz,
    z: B.Numeric,
    x,
    *,
    dtype_lik=None,
    **kw_args,
):
    if dtype_lik:
        z = B.cast(dtype_lik, z)
    return xz, Dirac(z, data_dims(xz))


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
    z = B.reshape(z, *shape_before, -1, *shape_after)
    return z, shape_compressed


def _split_dimension(z, axis, dims):
    """Split a dimension of a tensor into multiple dimensions.

    Args:
        z (tensor): Tensor to split.
        axis (int): Axis to split
        dims (iterable[int]): Sizes of new dimensions.

    Returns:
        tensor: Reshapes version of `z`.
    """
    shape = B.shape(z)
    # The indexing below will only be correct for positive `axis`, so resolve the index.
    axis = resolve_axis(z, axis)
    return B.reshape(z, *shape[:axis], *dims, *shape[axis + 1 :])


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
    d = data_dims(xz)
    dim_y = B.shape(z, -d - 1) // 2

    z_mean, z_noise = split_channels(z, (dim_y, dim_y), d)

    # Vectorise the data. Record the shape!
    z_mean, shape = _vectorise(z_mean, d + 1)
    z_noise, _ = _vectorise(z_noise, d + 1)

    # Transform into parameters.
    mean = z_mean
    noise = coder.epsilon + B.softplus(z_noise)

    # Cast parameters to the right data type.
    if dtype_lik:
        mean = B.cast(dtype_lik, mean)
        noise = B.cast(dtype_lik, noise)

    # Make a noiseless prediction.
    if noiseless:
        with B.on_device(noise):
            noise = B.zeros(noise)

    return xz, MultiOutputNormal.diagonal(mean, noise, shape)


@register_module
class LowRankGaussianLikelihood:
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
    d = data_dims(xz)
    dim_y = B.shape(z, -d - 1) // (2 + coder.rank)
    dim_inner = B.shape(z, -d - 1) - 2 * dim_y

    z_mean, z_noise, z_var_factor = split_channels(
        z, (dim_y, dim_y, dim_y * coder.rank), d
    )

    # Vectorise the data. Record the shape!
    z_mean, shape = _vectorise(z_mean, d + 1)
    z_noise, _ = _vectorise(z_noise, d + 1)
    # Split of the ranks of the factor.
    z_var_factor = _split_dimension(z_var_factor, -d - 1, (coder.rank, dim_y))
    z_var_factor, _ = _vectorise(z_var_factor, d + 1)
    # Put the dimensions of the factor last, because that it what the constructor
    # assumes.
    z_var_factor = B.transpose(z_var_factor)

    # Transform into parameters.
    mean = z_mean
    noise = coder.epsilon + B.softplus(z_noise)
    var_factor = z_var_factor / B.sqrt(B.shape(z_var_factor, -1))

    # Cast the parameters before constructing the distribution.
    if dtype_lik:
        mean = B.cast(dtype_lik, mean)
        noise = B.cast(dtype_lik, noise)
        var_factor = B.cast(dtype_lik, var_factor)

    # Make a noiseless prediction.
    if noiseless:
        with B.on_device(noise):
            noise = B.zeros(noise)

    return xz, MultiOutputNormal.lowrank(mean, noise, var_factor, shape)


@register_module
class DenseGaussianLikelihood:
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
        return "DenseGaussianLikelihood()"


@_dispatch
def code(
    coder: DenseGaussianLikelihood,
    xz: Parallel,
    z: Parallel,
    x,
    *,
    dtype_lik=None,
    noiseless=False,
    **kw_args,
):
    z1, z2 = z

    # Extract `d` and `dim_y` from the mean.
    d = data_dims(xz[0])
    dim_y = B.shape(z1, -d - 1) // 2

    z_mean, z_noise = split_channels(z1, (dim_y, dim_y), d)

    # Vectorise the data. Record the shape!
    z_mean, shape = _vectorise(z_mean, d + 1)
    z_noise, _ = _vectorise(z_noise, d + 1)
    # First vectorise inner channels and then vectorise outer channels.
    z_var, _ = _vectorise(z2, d + 1, offset=d + 1)
    z_var, _ = _vectorise(z_var, d + 1)

    # Transform into parameters.
    mean = z_mean
    noise = coder.epsilon + B.softplus(z_noise)
    var = z_var

    # Cast parameters to the right data type.
    if dtype_lik:
        mean = B.cast(dtype_lik, mean)
        noise = B.cast(dtype_lik, noise)
        var = B.cast(dtype_lik, var)

    # Make a noiseless prediction.
    if noiseless:
        with B.on_device(noise):
            noise = B.zeros(noise)

    # Just return the inputs for the mean.
    xz = xz[0]

    return xz, MultiOutputNormal.dense(mean, noise, var, shape)

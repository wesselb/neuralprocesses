import lab as B
import numpy as np
from matrix import Diagonal, LowRank, Woodbury
from plum import Union, parametric
from stheno import Normal
from wbml.util import indented_kv

from .dist import AbstractMultiOutputDistribution
from .. import _dispatch
from ..aggregate import Aggregate
from ..util import batch, split, register_module

__all__ = ["MultiOutputNormal", "DensifyLowRankVariance"]


@_dispatch
def _map_sample_output(f, state_res: tuple):
    state, res = state_res
    return state, f(res)


@_dispatch
def _map_sample_output(f, res):
    return f(res)


@_dispatch
def _monormal_vectorise(x: B.Numeric, shape):
    return B.reshape(x, *batch(x, len(shape)), -1, 1)


@_dispatch
def _monormal_vectorise(x: Aggregate, shape: Aggregate):
    return B.concat(*(_monormal_vectorise(xi, si) for xi, si in zip(x, shape)), axis=-2)


@_dispatch
def _monormal_unvectorise(x: B.Numeric, shape, *, squeeze_sample_dim=False):
    x = B.reshape(x, *B.shape(x)[:-1], *shape)
    if squeeze_sample_dim and B.shape(x, 0) == 1:
        x = x[0, ...]
    return x


@_dispatch
def _monormal_unvectorise(x: B.Numeric, shape: Aggregate, **kw_args):
    ns = [np.prod(si) for si in shape]
    xs = split(x, ns, -1)
    return Aggregate(
        *(_monormal_unvectorise(xi, si, **kw_args) for xi, si in zip(xs, shape))
    )


@parametric
class MultiOutputNormal(AbstractMultiOutputDistribution):
    """A normal distribution for multiple outputs. Use one of the class methods to
    construct the object.

    Args:
        normal (class:`stheno.Normal`): Underlying vectorised one-dimensional normal
            distribution.
        shape (shape or :class:`neuralprocesses.aggregate.Aggregate`): Shape(s) of the
            data before vectorising.

    Attributes:
        normal (class:`stheno.Normal`): Underlying vectorised one-dimensional normal
            distribution.
        shape (shape or :class:`neuralprocesses.aggregate.Aggregate`): Shape(s) of the
            data before vectorising.
    """

    @_dispatch
    def __init__(self, normal: Normal, shape):
        self.normal = normal
        self.shape = shape

    @classmethod
    def __infer_type_parameter__(cls, normal, shape):
        return (type(normal.mean), type(normal.var))

    @classmethod
    def dense(cls, mean: B.Numeric, var_diag: B.Numeric, var: B.Numeric, shape):

        """Construct a dense multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, n)`.
            var_diag (tensor): Marginal variances of shape `(*b, n)`.
            var (tensor): Variance matrix of shape `(*b, n, n)`.
            shape (shape): Shape of the data before vectorising.
        """
        return cls(Normal(mean[..., None], B.add(Diagonal(var_diag), var)), shape)

    @classmethod
    def diagonal(cls, mean: B.Numeric, var: B.Numeric, shape):
        """Construct a diagonal multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, n)`.
            var (tensor): Marginal variances of shape `(*b, n)`.
            shape (shape): Shape of the data before vectorising.
        """
        return cls(Normal(mean[..., None], Diagonal(var)), shape)

    @classmethod
    def lowrank(
        cls,
        mean: B.Numeric,
        var_diag: B.Numeric,
        var_factor: B.Numeric,
        shape,
        var_middle: Union[B.Numeric, None] = None,
    ):
        """Construct a low-rank multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, n)`.
            var_diag (tensor): Diagonal part of the low-rank variance of shape `(*b, n)`.
            var_factor (tensor): Factors of the low-rank variance of shape `(*b, n, f)`.
            shape (shape): Shape of the data before vectorising.
            var_middle (tensor, optional): Covariance of the factors of shape
                `(*b, f, f)`.
        """
        # Construct variance.
        var = Diagonal(var_diag) + LowRank(left=var_factor, middle=var_middle)
        return cls(Normal(mean[..., None], var), shape)

    def __repr__(self):
        return (  # Comment to preserve formatting.
            f"<MultiOutputNormal: shape={self.shape}\n"
            + indented_kv("normal", repr(self.normal), suffix=">")
        )

    def __str__(self):
        return (  # Comment to preserve formatting.
            f"<MultiOutputNormal: shape={self.shape}\n"
            + indented_kv("normal", str(self.normal), suffix=">")
        )

    @property
    def mean(self):
        return _monormal_unvectorise(self.normal.mean[..., 0], self.shape)

    @property
    def var(self):
        return _monormal_unvectorise(B.diag(self.normal.var), self.shape)

    def logpdf(self, x):
        return self.normal.logpdf(_monormal_vectorise(x, self.shape))

    def sample(self, *args, **kw_args):
        def f(sample):
            # Put the sample dimension first.
            perm = list(range(B.rank(sample)))
            perm = [perm[-1]] + perm[:-1]
            sample = B.transpose(sample, perm=perm)
            # Undo the vectorisation.
            sample = _monormal_unvectorise(sample, self.shape, squeeze_sample_dim=True)
            return sample

        return _map_sample_output(f, self.normal.sample(*args, **kw_args))

    def kl(self, other: "MultiOutputNormal"):
        return self.normal.kl(other.normal)

    def entropy(self):
        return self.normal.entropy()


@B.dispatch
def dtype(dist: MultiOutputNormal):
    return B.dtype(dist.normal)


@B.dispatch
def cast(dtype: B.DType, dist: MultiOutputNormal):
    return MultiOutputNormal(B.cast(dtype, dist.normal), dist.num_outputs)


@register_module
class DensifyLowRankVariance:
    """Densify a Woodbury variance if the low-rank part has more ranks than the
    dimensionality of the matrix."""

    @_dispatch
    def __call__(self, z: MultiOutputNormal):
        return z

    @_dispatch
    def __call__(self, z: MultiOutputNormal[B.Numeric, Woodbury]):
        if z.normal.var.lr.rank >= B.shape_matrix(z.normal.var, 0):
            return MultiOutputNormal(
                Normal(z.normal.mean, B.dense(z.normal.var)), z.shape
            )
        else:
            return z

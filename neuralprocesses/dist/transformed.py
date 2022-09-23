from functools import partial

import lab as B
from wbml.util import indented_kv

from .dist import AbstractDistribution, shape_batch
from .normal import _map_sample_output
from .. import _dispatch
from ..aggregate import Aggregate
from ..util import register_module

__all__ = ["Transform", "TransformedMultiOutputDistribution"]


@register_module
class Transform:
    """A transform for distributions.

    Args:
        transform (function): The transform.
        transform_derive (function): Derivative of the transform.
        untransform (function): Inverse of the transform.
        untransform_logdet (function): Log-determinant of the Jacobian of the inverse
            transform.

    Attributes:
        transform (function): The transform.
        transform_derive (function): Derivative of the transform.
        untransform (function): Inverse of the transform.
        untransform_logdet (function): Log-determinant of the Jacobian of the inverse
            transform.
    """

    def __init__(
        self,
        transform,
        transform_deriv,
        untransform,
        untransform_logdet,
    ):
        self.transform = transform
        self.transform_deriv = transform_deriv
        self.untransform = untransform
        self.untransform_logdet = untransform_logdet

    def __call__(self, dist):
        return TransformedMultiOutputDistribution(dist, self)

    @classmethod
    def exp(cls):
        """Construct the `exp` transform."""

        def transform(x):
            return B.exp(x)

        def transform_deriv(x):
            return B.exp(x)

        def untransform(y):
            return B.log(y)

        def untransform_logdet(y):
            return -B.log(y)

        return cls(
            transform=transform,
            transform_deriv=transform_deriv,
            untransform=untransform,
            untransform_logdet=untransform_logdet,
        )

    @classmethod
    def softplus(cls):
        """Construct the `softplus` transform."""

        def transform(x):
            return B.softplus(x)

        def transform_deriv(x):
            u = B.maximum(x, B.zero(x))
            x = x - u
            return B.exp(x) / (B.exp(x) + B.exp(-u))

        def untransform(y):
            # Clip the values to prevent NaNs
            y_clipped = B.minimum(B.maximum(y, B.exp(-20) * B.one(y)), 20 * B.one(y))
            # For big values, use an approximation.
            res = B.where(y > 20, y, B.log(B.exp(y_clipped) - 1))
            # For small values, also use an approximation.
            res = B.where(y < B.exp(-20), B.log(y), res)
            return res

        def untransform_logdet(y):
            # Use the same approximations as above.
            y_clipped = B.minimum(B.maximum(y, B.exp(-20) * B.one(y)), 20 * B.one(y))
            res = B.where(y > 20, B.zero(y), y_clipped - B.log(B.exp(y_clipped) - 1))
            res = B.where(y < B.exp(-20), -B.log(y), res)
            return res

        return cls(
            transform=transform,
            transform_deriv=transform_deriv,
            untransform=untransform,
            untransform_logdet=untransform_logdet,
        )

    @classmethod
    def bounded(cls, lower, upper):
        """Construct a transform for a bounded variable.

        Args:
            lower (scalar): Lower bound.
            upper (scalar): Upper bound.
        """

        def transform(x):
            return lower + (upper - lower) / (1 + B.exp(-x))

        def transform_deriv(x):
            denom = 1 + B.exp(-x)
            return (upper - lower) * B.exp(-x) / (denom * denom)

        def untransform(y):
            return B.log(y - lower) - B.log(upper - y)

        def untransform_logdet(y):
            return B.log(1 / (y - lower) + 1 / (upper - y))

        return cls(
            transform=transform,
            transform_deriv=transform_deriv,
            untransform=untransform,
            untransform_logdet=untransform_logdet,
        )

    @classmethod
    def signed_square(cls):
        """Construct the transform `f(x) = sign(x) |x|^2`."""

        def transform(x):
            return B.sign(x) * (x * x)

        def transform_deriv(x):
            return 2 * B.abs(x)

        def untransform(y):
            return B.sign(y) * B.sqrt(B.abs(y))

        def untransform_logdet(y):
            return -B.log(2) - B.log(B.abs(y))

        return cls(
            transform=transform,
            transform_deriv=transform_deriv,
            untransform=untransform,
            untransform_logdet=untransform_logdet,
        )


class TransformedMultiOutputDistribution(AbstractDistribution):
    """A transformed multi-output distribution.

    Args:
        dist (:class:`.AbstractMultiOutputDistribution`): Transformed distribution.
        transform (:class:`.Transform`): Transform.

    Attributes:
        dist (:class:`.AbstractMultiOutputDistribution`): Transformed distribution.
        transform (:class:`.Transform`): Transform.
        shape (shape or :class:`neuralprocesses.aggregate.Aggregate`): Shape(s) of the
            data before vectorising.
    """

    def __init__(self, dist, transform):
        self.dist = dist
        self.transform = transform

    @property
    def shape(self):
        """shape (shape or :class:`neuralprocesses.aggregate.Aggregate`): Shape(s) of
        the data before vectorising."""
        return self.dist.shape

    def __repr__(self):
        return (
            f"<TransformedMultiOutputDistribution:\n"
            + indented_kv("dist", repr(self.dist), suffix="\n")
            + indented_kv("transform", repr(self.transform), suffix=">")
        )

    def __str__(self):
        return (
            f"<TransformedMultiOutputDistribution:\n"
            + indented_kv("dist", str(self.dist), suffix="\n")
            + indented_kv("transform", str(self.transform), suffix=">")
        )

    @property
    def noiseless(self):
        """:class:`.TransformedMultiOutputNormal`: Noiseless version of the
        distribution."""
        return TransformedMultiOutputDistribution(self.dist.noiseless, self.transform)

    @property
    def mean(self):
        return _map_aggregate(self.transform.transform, self.dist.mean)

    @property
    def var(self):
        def _var(m, v):
            deriv = self.transform.transform_deriv(m)
            return deriv * deriv * v

        return _map_aggregate(_var, self.dist.mean, self.dist.var)

    def logpdf(self, x):
        def _logdet_sum(x, shape):
            return B.sum(
                self.transform.untransform_logdet(x),
                axis=tuple(range(-len(shape), 0)),
            )

        logpdf = self.dist.logpdf(_map_aggregate(self.transform.untransform, x))
        logdet = _sum_aggregate(_map_aggregate(_logdet_sum, x, self.shape))
        return logpdf + logdet

    def sample(self, *args, **kw_args):
        return _map_sample_output(
            partial(_map_aggregate, self.transform.transform),
            self.dist.sample(*args, **kw_args),
        )


@B.dispatch
def dtype(dist: TransformedMultiOutputDistribution):
    return B.dtype(dist.dist)


@B.dispatch
def cast(dtype: B.DType, dist: TransformedMultiOutputDistribution):
    return TransformedMultiOutputDistribution(B.cast(dtype, dist.dist), dist.transform)


@shape_batch.dispatch
def shape_batch(dist: TransformedMultiOutputDistribution):
    return shape_batch(dist.dist)


@_dispatch
def _map_aggregate(f, *xs):
    return f(*xs)


@_dispatch
def _map_aggregate(f, *xs: Aggregate):
    return Aggregate(*(f(*xis) for xis in zip(*xs)))


@_dispatch
def _sum_aggregate(x):
    return x


@_dispatch
def _sum_aggregate(x: Aggregate):
    return sum(x)

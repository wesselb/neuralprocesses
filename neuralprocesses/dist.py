import abc

import lab as B
from matrix import Diagonal, LowRank
from plum import Dispatcher, Union
from stheno import Normal
from wbml.util import indented_kv
from .util import batch

__all__ = ["Dirac", "MultiOutputNormal", "Transform"]

_dispatch = Dispatcher()


class AbstractMultiOutputDistribution(metaclass=abc.ABCMeta):
    """An interface for multi-output distributions."""

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def mean(self):
        """tensor: Marginal means of shape `(*b, c, n)`."""

    @abc.abstractmethod
    def var(self):
        """tensor: Marginal variances of shape `(*b, c, n)`."""

    @abc.abstractmethod
    def logpdf(self, x):
        """Compute the log-pdf at inputs `x`.

        Args:
            x (tensor): Inputs of shape `(*b, c, n)`.

        Returns:
            tensor: Log-pdfs of shape `(*b,)`.
        """

    @abc.abstractmethod
    def sample(self, state, num=1):
        """Sample from the distribution.

        Args:
            state (random state, optional): Random state.
            num (int, optional): Number of samples. Defaults to one.

        Returns:
            tensor: Samples of shape `(num, *b, c, n)` if `num > 1` and of shape .
                `(*b, c, n)` otherwise.
        """

    def kl(self, other):
        """Compute the KL-divergence with respect to another distribution.

        Args:
            other (:class:`.AbstractMultiOutputDistribution`): Other.

        Returns:
            tensor: KL-divergences `kl(self, other)` of shape `(*b,)`.
        """
        raise NotImplementedError(
            f"Cannot compute the KL-divergence between {self} and {other}."
        )

    def entropy(self):
        """Compute entropy.

        Returns:
            tensor: Entropies of shape `(*b,)`.
        """
        raise NotImplementedError(f"Cannot compute the entropy of {self}.")


class Dirac(AbstractMultiOutputDistribution):
    """A Dirac delta.

    Args:
        x (tensor): Position of the Dirac delta of shape `(*b, c, n)`.
    """

    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"<Dirac:\n" + indented_kv("x", repr(self.x), suffix=">")

    def __str__(self):
        return f"<Dirac:\n" + indented_kv("x", str(self.x), suffix=">")

    @property
    def mean(self):
        return self.x

    @property
    def var(self):
        with B.on_device(self.x):
            return B.zeros(x)

    def logpdf(self, x):
        with B.on_device(self.x):
            return B.zeros(B.dtype(self.x), *batch(x, 2))

    @_dispatch
    def sample(self, state: B.RandomState, num=1):
        return state, self.sample(num=num)

    @_dispatch
    def sample(self, num=1):
        if num == 1:
            return self.x
        else:
            return self.x[None, ...]

    def kl(self, other: "Dirac"):
        with B.on_device(self.x):
            return B.zeros(B.dtype(self.x), *batch(self.x, 2))


@_dispatch
def _map_sample_output(f, state_res: tuple):
    state, res = state_res
    return state, f(res)


@_dispatch
def _map_sample_output(f, res):
    return f(res)


class MultiOutputNormal(AbstractMultiOutputDistribution):
    """A normal distribution for multiple outputs. Use one of the class methods to
    construct the object.

    Args:
        normal (class:`stheno.Normal`): Underlying vectorised one-dimensional normal
            distribution.
        num_outputs (int): Number of outputs.

    Attributes:
        normal (class:`stheno.Normal`): Underlying vectorised one-dimensional normal
            distribution.
        num_outputs (int): Number of outputs.
    """

    @_dispatch
    def __init__(self, normal: Normal, num_outputs: B.Int):
        self.normal = normal
        self.num_outputs = num_outputs

    @classmethod
    def dense(cls, mean: B.Numeric, var: B.Numeric):
        """Construct a dense multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, c, n)`.
            var (tensor): Variance of shape `(*b, c, n, c, n)`.
        """
        c, n = B.shape(mean, -2, -1)
        return cls(
            Normal(
                B.reshape(mean, *batch(mean, 2), c * n, -1),
                B.reshape(var, *batch(var, 4), c * n, c * n),
            ),
            c,
        )

    @classmethod
    def diagonal(cls, mean: B.Numeric, var: B.Numeric):
        """Construct a diagonal multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, c, n)`.
            var (tensor): Marginal variances of shape `(*b, c, n)`.
        """
        c, n = B.shape(mean, -2, -1)
        return cls(
            Normal(
                B.reshape(mean, *batch(mean, 2), c * n, -1),
                Diagonal(B.reshape(var, *batch(var, 2), c * n)),
            ),
            c,
        )

    @classmethod
    def lowrank(
        cls,
        mean: B.Numeric,
        var_diag: B.Numeric,
        var_factor: B.Numeric,
        var_middle: Union[B.Numeric, None] = None,
    ):
        """Construct a low-rank multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, c, n)`.
            var_diag (tensor): Diagonal part of the low-rank variance of shape
                `(*b, c, n)`.
            var_factor (tensor): Factors of the low-rank variance of shape
                `(*b, c * num_factors, n)`.
            var_middle (tensor, optional): Covariance of the factors of shape
                `(*b, num_factors, num_factors)`.
        """
        c, n = B.shape(mean, -2, -1)
        # Separate out factor channels.
        var_factor = B.reshape(var_factor, *batch(var_factor, 2), c, -1, n)
        var_factor = B.transpose(var_factor)
        # Construct variance.
        var = Diagonal(B.reshape(var_diag, *batch(var_diag, 2), c * n))
        var += LowRank(
            left=B.reshape(var_factor, *batch(var_factor, 3), c * n, -1),
            middle=var_middle,
        )
        # Do not retain structure if it doesn't give computational savings.
        if var.lr.rank >= B.shape_matrix(var, 0):
            var = B.dense(var)
        return cls(Normal(B.reshape(mean, *batch(mean, 2), c * n, -1), var), c)

    def __repr__(self):
        return (  # Comment to preserve formatting.
            f"<MultiOutputNormal: num_outputs={self.num_outputs}\n"
            + indented_kv("normal", repr(self.normal), suffix=">")
        )

    def __str__(self):
        return (  # Comment to preserve formatting.
            f"<MultiOutputNormal: num_outputs={self.num_outputs}\n"
            + indented_kv("normal", str(self.normal), suffix=">")
        )

    def _unreshape(self, x):
        return B.reshape(x, *B.shape(x)[:-1], self.num_outputs, -1)

    @property
    def mean(self):
        return self._unreshape(self.normal.mean[..., 0])

    @property
    def var(self):
        return self._unreshape(B.diag(self.normal.var))

    def logpdf(self, x):
        return self.normal.logpdf(B.reshape(x, *batch(x, 2), -1, 1))

    def sample(self, *args, **kw_args):
        def f(sample):
            # Put the sample dimension first.
            perm = list(range(B.rank(sample)))
            perm = [perm[-1]] + perm[:-1]
            sample = B.transpose(sample, perm=perm)
            # Separate out outputs.
            sample = self._unreshape(sample)
            # If there was only one sample, squeeze the sample dimension.
            if B.shape(sample, 0) == 1:
                sample = sample[0, ...]
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
    def positive(cls):
        """Construct the `exp` transform."""

        def transform(x):
            return B.exp(x)

        def transform_deriv(x):
            return B.exp(x)

        def untransform(x):
            return B.log(x)

        def untransform_logdet(x):
            return -B.log(x)

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

        def untransform(x):
            return B.log(x - lower) - B.log(upper - x)

        def untransform_logdet(x):
            return B.log(1 / (x - lower) + 1 / (upper - x))

        return cls(
            transform=transform,
            transform_deriv=transform_deriv,
            untransform=untransform,
            untransform_logdet=untransform_logdet,
        )


class TransformedMultiOutputDistribution(AbstractMultiOutputDistribution):
    """A transformed multi-output distribution.

    Args:
        dist (:class:`.AbstractMultiOutputDistribution`): Transformed distribution.
        transform (:class:`.Transform`): Transform.

    Attributes:
        dist (:class:`.AbstractMultiOutputDistribution`): Transformed distribution.
        transform (:class:`.Transform`): Transform.
    """

    def __init__(self, dist, transform):
        self.dist = dist
        self.transform = transform

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
    def mean(self):
        return self.transform.transform(self.dist.mean)

    @property
    def var(self):
        deriv = self.transform.transform_deriv(self.dist.mean)
        return deriv * deriv * self.dist.var

    def logpdf(self, x):
        logdet = B.sum(self.transform.untransform_logdet(x), axis=(-2, -1))
        return self.dist.logpdf(self.transform.untransform(x)) + logdet

    def sample(self, *args, **kw_args):
        return _map_sample_output(
            self.transform.transform,
            self.dist.sample(*args, **kw_args),
        )


@B.dispatch
def dtype(dist: TransformedMultiOutputDistribution):
    return B.dtype(dist.dist)


@B.dispatch
def cast(dtype: B.DType, dist: TransformedMultiOutputDistribution):
    return TransformedMultiOutputDistribution(B.cast(dtype, dist.dist), dist.transform)

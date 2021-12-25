import lab as B
from matrix import Diagonal, LowRank
from plum import Dispatcher, Union
from stheno import Normal
from wbml.util import indented_kv

__all__ = ["MultiOutputNormal"]

_dispatch = Dispatcher()


class MultiOutputNormal:
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
            mean (tensor): Mean of shape `(b, c, n)`.
            var (tensor): Variance of shape `(b, c, n, c, n)`.
        """
        b, c, n = B.shape_matrix(mean)
        return cls(
            Normal(B.reshape(mean, b, c * n, -1), B.reshape(var, b, c * n, c * n)),
            c,
        )

    @classmethod
    def diagonal(cls, mean: B.Numeric, var: B.Numeric):
        """Construct a diagonal multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(b, c, n)`.
            var (tensor): Marginal variances of shape `(b, c, n)`.
        """
        b, c, n = B.shape(mean)
        return cls(
            Normal(B.reshape(mean, b, c * n, -1), Diagonal(B.reshape(var, b, c * n))),
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
            mean (tensor): Mean of shape `(b, c, n)`.
            var_diag (tensor): Diagonal part of the low-rank variance of shape
                `(b, c, n)`.
            var_factor (tensor): Factors of the low-rank variance of shape
                `(b, c * num_factors, n)`.
            var_middle (tensor, optional): Covariance of the factors of shape
                `(b, num_factors, num_factors)`.
        """
        b, c, n = B.shape(mean)
        # Separate out factor channels.
        var_factor = B.reshape(var_factor, b, c, -1, n)
        var_factor = B.transpose(var_factor)
        # Construct variance.
        var = Diagonal(B.reshape(var_diag, b, c * n))
        var = var + LowRank(left=B.reshape(var_factor, b, c * n, -1), middle=var_middle)
        # Do not retain structure if it doesn't give computational savings.
        if var.lr.rank >= B.shape_matrix(var, 0):
            var = B.dense(var)
        return cls(Normal(B.reshape(mean, b, c * n, -1), var), c)

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
        """tensor: Marginal means."""
        return self._unreshape(self.normal.mean[..., 0])

    @property
    def var(self):
        """tensor: Marginal variances."""
        return self._unreshape(B.diag(self.normal.var))

    def logpdf(self, x):
        """Compute the log-pdf at inputs `x`.

        Args:
            x (tensor): Inputs of shape `(b, c, n)`.

        Returns:
            tensor: Log-pdfs of shape `(b,)`.
        """
        return self.normal.logpdf(B.reshape(x, *B.shape(x)[:-2], -1, 1))

    def sample(self, *args, **kw_args):
        """Sample from the distribution.

        Args:
            state (random state, optional): Random state.
            num (int): Number of samples.
            noise (scalar, optional): Variance of noise to add to the
                samples. Must be positive.

        Returns:
            tensor: Samples of shape `(b, num, c, n)`.
        """
        res = self.normal.sample(*args, **kw_args)
        # Separate random state from `res`.
        if isinstance(res, tuple):
            state, res = res
        else:
            state = None
        # Reshape sample to have the right size.
        res = B.transpose(res)  # Put the sample dimension second to last.
        res = self._unreshape(res)
        # Also return random state in case it was separated off.
        if state is not None:
            return state, res
        else:
            return res

    def kl(self, other: "MultiOutputNormal"):
        """Compute the KL-divergence with respect to another multi-output normal.

        Args:
            other (MultiOutputNormal): Other.

        Returns:
            tensor: KL-divergences `kl(self, other)` of shape `(b,)`.
        """
        return self.normal.kl(other.normal)

    def entropy(self):
        """Compute entropy.

        Returns:
            tensor: Entropies of shape `(b,)`.
        """
        return self.normal.entropy()


@B.dispatch
def dtype(dist: MultiOutputNormal):
    return B.dtype(dist.normal)


@B.dispatch
def cast(dtype: B.DType, dist: MultiOutputNormal):
    return MultiOutputNormal(B.cast(dtype, dist.normal), dist.num_outputs)

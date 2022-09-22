import lab as B
from wbml.util import indented_kv

from .dist import AbstractDistribution
from .. import _dispatch

__all__ = ["SpikesSlab"]


class SpikesSlab(AbstractDistribution):
    """Spikes-and-slab distribution.

    Args:
        spikes (vector): Spikes.
        slab (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Slab.
        logprobs (tensor): Log-probabilities for the spikes and the slab with the
            log-probability for the slab last.
        epsilon (float, optional): Tolerance for equality checking. Defaults to `1e-6`.
        slab_safe_value (float, optional): Safe value to to evaluate the slab at.
            Defaults to `0.5`.

    Attributes:
        spikes (vector): Spikes.
        slab (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Slab.
        logprobs (tensor): Log-probabilities for the spikes and the slab with the
            log-probability for the slab last.
        epsilon (float): Tolerance for equality checking.
        slab_safe_value (float): Safe value to to evaluate the slab at.
    """

    @_dispatch
    def __init__(
        self,
        spikes: B.Numeric,
        slab: AbstractDistribution,
        logprobs: B.Numeric,
        *,
        epsilon: float = 1e-6,
        slab_safe_value: float = 0.5,
    ):
        self.spikes = spikes
        self.slab = slab
        # Normalise the probabilities to ensure that the log-pdf computation is correct.
        self.logprobs = logprobs - B.logsumexp(logprobs, axis=-1, squeeze=False)
        self.epsilon = epsilon
        self.slab_safe_value = slab_safe_value

    def __repr__(self):
        return (  # Comment to preserve formatting.
            f"<SpikeAndSlab:"
            f" spikes={self.spikes!r}"
            f" epsilon={self.epsilon!r}"
            f" slab_safe_value={self.slab_safe_value!r}\n"
            + indented_kv("slab", repr(self.slab), suffix=">")
            + indented_kv("logprobs", repr(self.slab), suffix=">")
        )

    def __str__(self):
        return (  # Comment to preserve formatting.
            f"<MultiOutputNormal:"
            f" spikes={self.spikes}\n"
            f" epsilon={self.epsilon}"
            f" slab_safe_value={self.slab_safe_value}\n"
            + indented_kv("slab", str(self.slab), suffix=">")
            + indented_kv("logprobs", str(self.slab), suffix=">")
        )

    def logpdf(self, x):
        # Construct indicators which filter out the spikes.
        spikes_indicator = B.cast(
            B.dtype(x),
            B.abs(
                B.expand_dims(self.spikes, axis=0, times=B.rank(x))
                - B.expand_dims(x, axis=-1, times=1)
            )
            < self.epsilon,
        )
        # Construct an indicator for the slab.
        with B.on_device(x):
            any_spike = B.minimum(B.sum(spikes_indicator, axis=-1), B.one(x))
            slab_indicator = B.one(x) - any_spike
        # Compute the log-probability of the categorical random variable.
        full_indicator = B.concat(spikes_indicator, slab_indicator[..., None], axis=-1)
        cat_logprob = B.sum(full_indicator * self.logprobs, axis=-1)
        # Compute the log-probability of the slab.
        with B.on_device(x):
            safe = B.to_active_device(B.cast(B.dtype(x), self.slab_safe_value))
            slab_logprob = slab_indicator * self.slab.logpdf(
                slab_indicator * x + safe * (B.one(x) - slab_indicator)
            )
        # Assemble the full log-probability and return.
        return cat_logprob + slab_logprob

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
        with B.on_device(self.logprobs):
            # Sample the categorical variable.
            state, inds = B.randcat(state, B.exp(self.logprobs), *shape)

            # Construct indicators to filter the spikes or slab samples.
            spikes_and_zero = B.concat(self.spikes, B.zero(self.spikes)[None])
            zeros_and_one = B.concat(B.zeros(self.spikes), B.one(self.spikes)[None])
            spikes = B.reshape(
                B.take(spikes_and_zero, B.flatten(inds)),
                *B.shape(inds),
            )
            slab_indicator = B.reshape(
                B.take(zeros_and_one, B.flatten(inds)),
                *B.shape(inds),
            )

            # Sample the slab everywhere. The indicators will pick the right samples.
            state, slab_sample = self.slab.sample(state, dtype, *shape)

        # Assemble the sample.
        sample = spikes + slab_indicator * slab_sample

        return state, sample


@B.dtype.dispatch
def dtype(d: SpikesSlab):
    return B.dtype(d.spikes, d.slab, d.logprobs)


@B.shape.dispatch
def shape(d: SpikesSlab):
    return B.shape(d.slab, d.logprobs[..., 0])

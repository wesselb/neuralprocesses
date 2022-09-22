import lab as B
from wbml.util import indented_kv

from .dist import AbstractDistribution
from .. import _dispatch

__all__ = ["SpikesAndSlab"]


class SpikesAndSlab(AbstractDistribution):
    """Spikes-and-slab distribution.

    Args:
        spikes (vector): Spikes.
        slab (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Slab.
        logprobs (tensor): Log-probabilities for the spikes and the slab with the
            log-probability for the slab last.

    Attributes:
        spikes (vector): Spikes.
        slab (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Slab.
        logprobs (tensor): Log-probabilities for the spikes and the slab with the
            log-probability for the slab last.
    """

    @_dispatch
    def __init__(
        self, spikes: B.Numeric, slab: AbstractDistribution, logprobs: B.Numeric
    ):
        self.spikes = spikes
        self.slab = slab
        self.logprobs = logprobs

    def __repr__(self):
        return (  # Comment to preserve formatting.
            f"<SpikeAndSlab: spikes={self.spikes!r}\n"
            + indented_kv("slab", repr(self.slab), suffix=">")
        )

    def __str__(self):
        return (  # Comment to preserve formatting.
            f"<MultiOutputNormal: spikes={self.spikes}\n"
            + indented_kv("slab", str(self.slab), suffix=">")
        )

    def logpdf(self, x):
        pass

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
        shape = shape + B.shape(self.logprobs)[:-1]

        with B.on_device(self.logprobs):
            # Sample the categorical variable.
            state, inds = B.randcat(state, B.exp(self.logprobs), *shape)

            # Construct indicators to filter the spikes or slab samples.
            spikes_and_zero = B.concat(self.spikes, B.zero(self.spikes)[None])
            zeros_and_one = B.concat(B.zeros(self.spikes), B.one(self.spikes)[None])
            spikes = B.reshape(B.take(spikes_and_zero, B.flatten(inds)), *shape)
            slab_indicator = B.reshape(B.take(zeros_and_one, B.flatten(inds)), *shape)

            # Sample the slab everywhere. The indicators will pick the right samples.
            state, slab_sample = self.slab.sample(state, dtype, *shape)

        # Assemble the sample.
        sample = spikes + slab_indicator * slab_sample

        return state, sample


@B.dtype.dispatch
def dtype(d: SpikesAndSlab):
    return B.dtype(d.spikes, d.slab, d.logprobs)


@B.shape.dispatch
def shape(d: SpikesAndSlab):
    return B.shape(d.slab, d.logprobs[..., 0])

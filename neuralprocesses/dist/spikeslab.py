import lab as B
from matrix.shape import broadcast
from wbml.util import indented_kv

from .dist import AbstractDistribution
from .. import _dispatch
from ..aggregate import Aggregate

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
        spikes,
        slab: AbstractDistribution,
        logprobs,
        d,
        *,
        epsilon: float = 1e-6,
        slab_safe_value: float = 0.5,
    ):
        self.spikes = spikes
        self.slab = slab
        # Normalise the probabilities to ensure that the log-pdf computation is correct.
        self.logprobs = B.subtract(
            logprobs,
            B.logsumexp(logprobs, axis=-1, squeeze=False),
        )
        self.d = d
        self.epsilon = epsilon
        self.slab_safe_value = slab_safe_value

    def __repr__(self):
        return (  # Comment to preserve formatting.
            f"<SpikesSlab:"
            f" spikes={self.spikes!r},"
            f" epsilon={self.epsilon!r},"
            f" slab_safe_value={self.slab_safe_value!r}\n"
            + indented_kv("slab", repr(self.slab), suffix="\n")
            + indented_kv("logprobs", repr(self.logprobs), suffix=">")
        )

    def __str__(self):
        return (  # Comment to preserve formatting.
            f"<SpikeSlab:"
            f" spikes={self.spikes},"
            f" epsilon={self.epsilon},"
            f" slab_safe_value={self.slab_safe_value}\n"
            + indented_kv("slab", str(self.slab), suffix="\n")
            + indented_kv("logprobs", str(self.logprobs), suffix=">")
        )

    @property
    def mean(self):
        return self.m1

    @property
    def var(self):
        m1 = self.m1
        return B.subtract(self.m2, B.multiply(m1, m1))

    @property
    def m1(self):
        m1_sp = self.spikes
        m1_sl = self.slab.m1
        m1 = _spikeslab_concat(m1_sp, m1_sl)
        return B.sum(B.multiply(B.exp(self.logprobs), m1), axis=-1)

    @property
    def m2(self):
        m2_sp = self.spikes * self.spikes
        m2_sl = self.slab.m2
        m2 = _spikeslab_concat(m2_sp, m2_sl)
        return B.sum(B.multiply(B.exp(self.logprobs), m2), axis=-1)

    @_dispatch
    def logpdf(self, x):
        logprob_cat, ind_slab = self.logpdf_cat(self.logprobs, x)
        x = self.logpdf_adjust_inputs(ind_slab, x)
        logpdf_slab = self.slab.logpdf(x)
        logpdf = B.add(logprob_cat, B.multiply(ind_slab, logpdf_slab))
        logpdf = _sum_sample_dims(logpdf, self.d)
        return _sum_aggregate(logpdf)

    @_dispatch
    def logpdf_cat(self, logprobs: Aggregate, x: Aggregate):
        li, ii = zip(*(self.logpdf_cat(li, xi) for li, xi in zip(logprobs, x)))
        return Aggregate(*li), Aggregate(*ii)

    @_dispatch
    def logpdf_cat(self, logprobs: B.Numeric, x: B.Numeric):
        # Construct indicators which filter out the spikes.
        ind_spikes = B.cast(
            B.dtype(x),
            B.abs(
                B.expand_dims(self.spikes, axis=0, times=B.rank(x))
                - B.expand_dims(x, axis=-1, times=1)
            )
            < self.epsilon,
        )
        # Construct an indicator for the slab.
        with B.on_device(x):
            any_spike = B.minimum(B.sum(ind_spikes, axis=-1), B.one(x))
            ind_slab = B.one(x) - any_spike
        # Compute the log-probability of the categorical random variable.
        full_indicator = B.concat(ind_spikes, ind_slab[..., None], axis=-1)
        logprob_cat = B.sum(full_indicator * logprobs, axis=-1)
        return logprob_cat, ind_slab

    @_dispatch
    def logpdf_adjust_inputs(self, ind_slab: Aggregate, x: Aggregate):
        return Aggregate(
            *(self.logpdf_adjust_inputs(ii, xi) for ii, xi in zip(ind_slab, x))
        )

    @_dispatch
    def logpdf_adjust_inputs(self, ind_slab: B.Numeric, x: B.Numeric):
        with B.on_device(x):
            safe = B.to_active_device(B.cast(B.dtype(x), self.slab_safe_value))
            return ind_slab * x + safe * (B.one(x) - ind_slab)

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
        state, slab_sample = self.slab.sample(state, dtype, *shape)
        return self.sample_spikes(
            self.logprobs,
            slab_sample,
            state,
            dtype,
            *shape,
        )

    @_dispatch
    def sample_spikes(
        self,
        logprobs: Aggregate,
        sample_slab: Aggregate,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        samples = []
        for li, si in zip(logprobs, sample_slab):
            state, sample = self.sample_spikes(li, si, state, dtype, *shape)
            samples.append(sample)
        return state, Aggregate(*samples)

    @_dispatch
    def sample_spikes(
        self,
        logprobs: B.Numeric,
        sample_slab: B.Numeric,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        with B.on_device(logprobs):
            # Sample the categorical variable.
            state, inds = B.randcat(state, B.exp(logprobs), *shape)

            # Construct indicators to filter the spikes or slab samples.
            spikes_and_zero = B.concat(self.spikes, B.zero(self.spikes)[None])
            zeros_and_one = B.concat(B.zeros(self.spikes), B.one(self.spikes)[None])
            sample_spikes = B.reshape(
                B.take(spikes_and_zero, B.flatten(inds)),
                *B.shape(inds),
            )
            slab_indicator = B.reshape(
                B.take(zeros_and_one, B.flatten(inds)),
                *B.shape(inds),
            )
        # Assemble sample.
        return state, sample_spikes + slab_indicator * sample_slab


@B.shape_batch.dispatch
def shape_batch(d: SpikesSlab):
    shape = B.shape_broadcast(d.slab, d.logprobs[..., 0])
    if isinstance(shape, Aggregate):
        shape = broadcast(*shape)
    return shape


@_dispatch
def _sum_sample_dims(x: B.Numeric, d: B.Int):
    for _ in range(d):
        x = B.sum(x, axis=-1)
    return x


@_dispatch
def _sum_sample_dims(x: Aggregate, d: Aggregate):
    return Aggregate(*(_sum_sample_dims(xi, di) for xi, di in zip(x, d)))


@_dispatch
def _sum_aggregate(x):
    return x


@_dispatch
def _sum_aggregate(x: Aggregate):
    return sum(x, 0)


@_dispatch
def _spikeslab_concat(spikes, slab):
    return B.concat(
        B.repeat(spikes, *B.shape(slab)),
        B.expand_dims(slab, axis=-1),
        axis=-1,
    )


@_dispatch
def _spikeslab_concat(spikes, slab: Aggregate):
    return Aggregate(*(_spikeslab_concat(spikes, si) for si in slab))

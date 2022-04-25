import lab as B
import numpy as np
from matrix import Diagonal
from matrix.util import indent
from plum import List, Tuple, Union
from stheno import Normal

from . import _dispatch
from .augment import AugmentedInput
from .coding import code, code_track, recode_stochastic
from .dist import AbstractMultiOutputDistribution, MultiOutputNormal
from .mask import Masked
from .parallel import Parallel
from .util import register_module

__all__ = ["Model", "loglik", "elbo", "predict"]


@register_module
class Model:
    """Encoder-decoder model.

    Args:
        encoder (coder): Coder.
        decoder (coder): Coder.

    Attributes:
        encoder (coder): Coder.
        decoder (coder): Coder.
    """

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    @_dispatch
    def __call__(
        self,
        state: B.RandomState,
        xc,
        yc,
        xt,
        *,
        num_samples=1,
        aux_t=None,
        dtype_enc_sample=None,
        **kw_args,
    ):
        """Run the model.

        Args:
            state (random state, optional): Random state.
            xc (input): Context inputs.
            yc (tensor): Context outputs.
            xt (input): Target inputs.
            num_samples (int, optional): Number of samples, if applicable. Defaults
                to 1.
            aux_t (tensor, optional): Target-specific auxiliary input, if applicable.
            dtype_enc_sample (dtype, optional): Data type to convert the sampled
                encoding to.

        Returns:
            random state, optional: Random state.
            input: Target inputs.
            object: Prediction for target outputs.
        """
        # Perform augmentation of `xt` with auxiliary target information.
        if aux_t is not None:
            xt = AugmentedInput(xt, aux_t)

        # If the keyword `noiseless` is set to `True`, then that only applies to the
        # decoder.
        enc_kw_args = dict(kw_args)
        if "noiseless" in enc_kw_args:
            del enc_kw_args["noiseless"]
        xz, pz = code(self.encoder, xc, yc, xt, **enc_kw_args)

        # Sample and convert sample to the right data type.
        state, z = _sample(state, pz, num=num_samples)
        if dtype_enc_sample:
            z = B.cast(dtype_enc_sample, z)

        _, d = code(self.decoder, xz, z, xt, **kw_args)

        return state, d

    @_dispatch
    def __call__(self, xc, yc, xt, **kw_args):
        state = B.global_random_state(B.dtype(xt))
        state, d = self(state, xc, yc, xt, **kw_args)
        B.set_global_random_state(state)
        return d

    @_dispatch
    def __call__(
        self,
        state: B.RandomState,
        contexts: List[Tuple[Union[B.Numeric, tuple], Union[B.Numeric, Masked]]],
        xt,
        **kw_args,
    ):
        return self(
            state,
            Parallel(*(c[0] for c in contexts)),
            Parallel(*(c[1] for c in contexts)),
            xt,
            **kw_args,
        )

    @_dispatch
    def __call__(
        self,
        contexts: List[Tuple[Union[B.Numeric, tuple], Union[B.Numeric, Masked]]],
        xt,
        **kw_args,
    ):
        state = B.global_random_state(B.dtype(xt))
        state, d = self(state, contexts, xt, **kw_args)
        B.set_global_random_state(state)
        return d

    def __str__(self):
        return (
            f"Model(\n"
            + indent(str(self.encoder), " " * 4)
            + ",\n"
            + indent(str(self.decoder), " " * 4)
            + "\n)"
        )

    def __repr__(self):
        return (
            f"Model(\n"
            + indent(repr(self.encoder), " " * 4)
            + ",\n"
            + indent(repr(self.decoder), " " * 4)
            + "\n)"
        )


@_dispatch
def _sample(state: B.RandomState, x: AbstractMultiOutputDistribution, num: B.Int = 1):
    return x.sample(state, num=num)


@_dispatch
def _sample(state: B.RandomState, x: Parallel, num: B.Int = 1):
    samples = []
    for xi in x:
        state, sample = _sample(state, xi, num=num)
        samples.append(sample)
    return state, Parallel(*samples)


@_dispatch
def _fix_noise(d: MultiOutputNormal, epoch: Union[int, None]):
    if epoch is not None and epoch < 3:
        # Fix noise to `1e-4`.
        var_diag = d.normal.var_diag
        with B.on_device(var_diag):
            var = Diagonal(1e-4 * B.ones(var_diag))
        d = MultiOutputNormal(Normal(d.normal.mean, var), d.shape)
    return d


@_dispatch
def loglik(
    state: B.RandomState,
    model: Model,
    xc,
    yc,
    xt,
    yt,
    *,
    num_samples=1,
    batch_size=256,
    normalise=True,
    epoch=None,
    **kw_args,
):
    """Log-likelihood objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (tensor): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (tensor): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        batch_size (int, optional): Batch size to use for sampling. Defaults to 256.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `True`.
        epoch (int, optional): Current epoch. If it is given, the likelihood variance
            is fixed to `1e-4` for the first three epochs to encourage the model to fit.

    Returns:
        random state, optional: Random state.
        tensor: Log-likelihoods.
    """
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)

    # If `num_samples = 1`, then there will not be a sample dimension, so we can
    # avoid the `logsumexp`.
    do_logsumexp = num_samples > 1

    # Sample in batches to alleviate memory requirements.
    logpdfs = None
    done_num_samples = 0
    while done_num_samples < num_samples:
        # Limit the number of samples at the batch size.
        this_num_samples = min(num_samples - done_num_samples, batch_size)

        # Perform batch.
        state, pred = model(
            state,
            xc,
            yc,
            xt,
            num_samples=this_num_samples,
            dtype_enc_sample=float,
            dtype_lik=float64,
            **kw_args,
        )
        pred = _fix_noise(pred, epoch)
        this_logpdfs = pred.logpdf(B.cast(float64, yt))

        # If the number of samples is equal to one but `num_samples > 1`, then the
        # likelihood was a `Dirac`, so we can stop batching. Also, set `num_samples = 1`
        # because we only have one sample now.
        if num_samples > 1 and B.shape(this_logpdfs, 0) == 1:
            logpdfs = this_logpdfs
            num_samples = 1
            break

        # Record current samples.
        if logpdfs is None:
            logpdfs = this_logpdfs
        else:
            # Concatenate at the sample dimension.
            logpdfs = B.concat(logpdfs, this_logpdfs, axis=0)

        # Increase the counter.
        done_num_samples += this_num_samples

    # Average over samples.
    if do_logsumexp:
        # Sample dimension should always be the first.
        logpdfs = B.logsumexp(logpdfs, axis=0) - B.log(num_samples)

    if normalise:
        # Normalise by the number of targets.
        logpdfs = logpdfs / B.shape(xt, -1)

    return state, logpdfs


@_dispatch
def loglik(
    model: Model,
    xc,
    yc,
    xt,
    yt,
    **kw_args,
):
    state = B.global_random_state(B.dtype(xt))
    state, logpdfs = loglik(state, model, xc, yc, xt, yt, **kw_args)
    B.set_global_random_state(state)
    return logpdfs


@_dispatch
def elbo(
    state: B.RandomState,
    model: Model,
    xc,
    yc,
    xt,
    yt,
    *,
    num_samples=1,
    normalise=True,
    subsume_context=True,
    epoch=None,
    **kw_args,
):
    """ELBO objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (tensor): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (tensor): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `True`.
        subsume_context (bool, optional): Subsume the context set into the target set.
            Defaults to `True`.
        epoch (int, optional): Current epoch. If it is given, the likelihood variance
            is fixed to `1e-4` for the first three epochs to encourage the model to fit.

    Returns:
        random state, optional: Random state.
        tensor: ELBOs.
    """
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)

    if subsume_context:
        # Subsume the context set into the target set.
        xt = B.concat(xc, xt, axis=-1)
        yt = B.concat(yc, yt, axis=-1)

    # Construct prior.
    xz, pz, h = code_track(model.encoder, xc, yc, xt, dtype_lik=float64, **kw_args)

    # Construct posterior.
    qz = recode_stochastic(model.encoder, pz, xt, yt, h, dtype_lik=float64, **kw_args)

    # Sample from poster.
    state, z = _sample(state, qz, num=num_samples)
    z = B.cast(float, z)

    # Run sample through decoder.
    _, d = code(model.decoder, xz, z, xt, dtype_lik=float64, **kw_args)
    d = _fix_noise(d, epoch)

    # Compute the ELBO.
    elbos = B.mean(d.logpdf(B.cast(float64, yt)), axis=0) - _kl(qz, pz)

    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.shape(xt, -1)

    return state, elbos


@_dispatch
def elbo(
    model: Model,
    xc,
    yc,
    xt,
    yt,
    **kw_args,
):
    state = B.global_random_state(B.dtype(xt))
    state, elbos = elbo(state, model, xc, yc, xt, yt, **kw_args)
    B.set_global_random_state(state)
    return elbos


@_dispatch
def _kl(q: AbstractMultiOutputDistribution, p: AbstractMultiOutputDistribution):
    return q.kl(p)


@_dispatch
def _kl(q: Parallel, p: Parallel):
    return sum([_kl(qi, pi) for qi, pi in zip(q, p)], 0)


def predict(model, xc, yc, xt, pred_num_samples=50, num_samples=5):
    """Use a model to predict.

    Args:
        model (:class:`.Model`): Model.
        xc (tensor): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (tensor): Inputs of the target set.
        pred_num_samples (int, optional): Number of samples to use for prediction.
            Defaults to 50.
        num_samples (int, optional): Number of noiseless samples to produce. Defaults
            to 5.

    Returns:
        tensor: Marignal mean.
        tensor: Marginal variance.
        tensor: `num_samples` noiseless samples.
    """
    float = B.dtype_float(xt)
    float64 = B.promote_dtypes(float, np.float64)

    # Predict marginal statistics.
    m1s, m2s = [], []
    for _ in range(pred_num_samples):
        pred = model(xc, yc, xt)
        m1s.append(pred.mean)
        m2s.append(pred.var + pred.mean**2)
    m1 = B.mean(B.stack(*m1s, axis=0), axis=0)
    m2 = B.mean(B.stack(*m2s, axis=0), axis=0)
    mean, var = m1, m2 - m1**2

    # Produce noiseless samples.
    samples = []
    for _ in range(num_samples):
        pred_noiseless = model(
            xc,
            yc,
            xt,
            dtype_enc_sample=float,
            dtype_lik=float64,
            noiseless=True,
        )
        # Try sampling with increasingly higher regularisation.
        epsilon_before = B.epsilon
        while True:
            try:
                samples.append(pred_noiseless.sample())
                break
            except Exception as e:
                B.epsilon *= 10
                if B.epsilon > 1e-3:
                    # Reset regularisation before failing.
                    B.epsilon = epsilon_before
                    raise e
        B.epsilon = epsilon_before  # Reset regularisation after success.
    samples = B.stack(*samples, axis=0)

    return mean, var, samples

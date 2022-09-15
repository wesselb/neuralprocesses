import lab as B
import numpy as np

from .model import Model
from .util import fix_noise as fix_noise_in_pred
from .. import _dispatch
from ..numdata import num_data

__all__ = ["loglik"]


@_dispatch
def loglik(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt,
    yt,
    *,
    num_samples=1,
    batch_size=16,
    normalise=False,
    fix_noise=None,
    **kw_args,
):
    """Log-likelihood objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        batch_size (int, optional): Batch size to use for sampling. Defaults to 16.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.

    Returns:
        random state, optional: Random state.
        tensor: Log-likelihoods.
    """
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)

    # Sample in batches to alleviate memory requirements.
    logpdfs = None
    done_num_samples = 0
    while done_num_samples < num_samples:
        # Limit the number of samples at the batch size.
        this_num_samples = min(num_samples - done_num_samples, batch_size)

        # Perform batch.
        state, pred = model(
            state,
            contexts,
            xt,
            num_samples=this_num_samples,
            dtype_enc_sample=float,
            dtype_lik=float64,
            **kw_args,
        )
        pred = fix_noise_in_pred(pred, fix_noise)
        this_logpdfs = pred.logpdf(B.cast(float64, yt))

        # If the number of samples is equal to one but `num_samples > 1`, then the
        # encoding was a `Dirac`, so we can stop batching. Also, set `num_samples = 1`
        # because we only have one sample now. We also don't need to do the
        # `logsumexp` anymore.
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

    # Average over samples. Sample dimension should always be the first.
    logpdfs = B.logsumexp(logpdfs, axis=0) - B.log(num_samples)

    if normalise:
        # Normalise by the number of targets.
        logpdfs = logpdfs / B.cast(float64, num_data(xt, yt))

    return state, logpdfs


@_dispatch
def loglik(state: B.RandomState, model: Model, xc, yc, xt, yt, **kw_args):
    return loglik(state, model, [(xc, yc)], xt, yt, **kw_args)


@_dispatch
def loglik(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-2]))
    state, logpdfs = loglik(state, model, *args, **kw_args)
    B.set_global_random_state(state)
    return logpdfs

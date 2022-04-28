import lab as B
import numpy as np
from wbml.util import inv_perm

from . import _dispatch
from .model import Model

__all__ = ["ar_predict", "ar_loglik"]


def _sort_targets(state, xt, yt=None, *, order):
    # Copy the targets because we'll modify them in-place.
    xt = B.identity(xt)
    if yt is not None:
        yt = B.identity(yt)

    # Sort the targets.
    inv_perms = []
    for i in range(B.shape(xt, 0)):
        # Determine ordering for task.
        if order == "left-to-right":
            perm = np.lexsort(B.to_numpy(xt)[i, ::-1, :])
        elif order == "random":
            state, perm = B.randperm(state, B.dtype_int(xt), B.shape(xt, -1))
        else:
            raise RuntimeError(f'Invalid ordering "{order}".')
        inv_perms.append(inv_perm(perm))

        xt[..., i, :, :] = B.take(xt[..., i, :, :], perm, axis=-1)
        if yt is not None:
            yt[..., i, :, :] = B.take(yt[..., i, :, :], perm, axis=-1)

    def unsort(z):
        """Undo the sorting."""
        z = B.identity(z)  # Make a copy, because we'll modify it.
        for i, perm in enumerate(inv_perms):
            z[..., i, :, :] = B.take(z[..., i, :, :], perm, axis=-1)
        return z

    if yt is None:
        return state, xt, unsort
    else:
        return state, xt, yt, unsort


@_dispatch
def ar_predict(
    state: B.RandomState,
    model: Model,
    xc,
    yc,
    xt,
    pred_num_samples=50,
    num_samples=5,
    order="random",
):
    """Autoregressive sampling.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (tensor): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (tensor): Inputs of the target set.
        pred_num_samples (int, optional): Number of samples to use for prediction.
            Defaults to 50.
        num_samples (int, optional): Number of noiseless samples to produce. Defaults
            to 5.
        order (str, optional): Order. Must be one of `"random"` or `"left-to-right"`.
            Defaults to `"random"`.

    Returns:
        random state, optional: Random state.
        tensor: Marginal mean.
        tensor: Marginal variance.
        tensor: `num_samples` noiseless samples.
    """
    # Perform sorting.
    state, xt, unsort = _sort_targets(state, xt, order=order)

    def _tile(x):
        return B.tile(x[None, ...], pred_num_samples, *((1,) * B.rank(x)))

    # Tile to produce multiple samples through batching.
    xc = _tile(xc)
    yc = _tile(yc)
    xt = _tile(xt)

    # Evaluate the model autoregressively.
    preds, samples = [], []
    for i in range(B.shape(xt, -1)):
        state, pred = model(state, xc, yc, xt[..., i : i + 1])
        state, sample = pred.sample(state)
        preds.append(pred)
        samples.append(sample)
        xc = B.concat(xc, xt[..., i : i + 1], axis=-1)
        yc = B.concat(yc, sample, axis=-1)
    samples = B.concat(*samples, axis=-1)

    # Produce predictive statistics.
    m1 = B.mean(B.concat(*(p.mean for p in preds), axis=-1), axis=0)
    m2 = B.mean(B.concat(*(p.var + p.mean**2 for p in preds), axis=-1), axis=0)
    mean, var = m1, m2 - m1**2

    # Produce noiseless samples.
    state, pred = model(
        state,
        xt[:num_samples, ...],
        samples[:num_samples, ...],
        xt[:num_samples, ...],
    )
    noiseless_samples = pred.mean

    return state, unsort(mean), unsort(var), unsort(noiseless_samples)


@_dispatch
def ar_predict(model: Model, xc, yc, xt, **kw_args):
    state = B.global_random_state(B.dtype(xt))
    state, mean, var, noiseless_samples = ar_predict(
        state, model, xc, yc, xt, **kw_args
    )
    return mean, var, noiseless_samples


@_dispatch
def ar_loglik(
    state: B.RandomState,
    model: Model,
    xc,
    yc,
    xt,
    yt,
    normalise=True,
    order="random",
):
    """Autoregressive log-likelihood.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (tensor): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (tensor): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `True`.
        order (str, optional): Order. Must be one of `"random"` or `"left-to-right"`.
            Defaults to `"random"`.

    Returns:
        random state, optional: Random state.
        tensor: Log-likelihoods.
    """
    state, xt, yt, _ = _sort_targets(state, xt, yt, order=order)

    # Now evaluate the log-likelihood autoregressively.
    logpdfs = []
    for i in range(B.shape(xt, -1)):
        pred = model(xc, yc, xt[..., i : i + 1])
        logpdfs.append(pred.logpdf(yt[..., i : i + 1]))
        xc = B.concat(xc, xt[..., i : i + 1], axis=-1)
        yc = B.concat(yc, yt[..., i : i + 1], axis=-1)
    logpdf = sum(logpdfs)

    if normalise:
        # Normalise by the number of targets.
        logpdf = logpdf / B.shape(xt, -1)

    return state, logpdf


@_dispatch
def ar_loglik(model: Model, xc, yc, xt, yt, **kw_args):
    state = B.global_random_state(B.dtype(xt))
    state, res = ar_loglik(state, model, xc, yc, xt, yt, **kw_args)
    return res

import lab as B
import numpy as np
from wbml.util import inv_perm

__all__ = ["ar_predict", "ar_loglik"]


def _sort_targets(xt, yt=None):
    # Copy the targets because we'll modify them in-place.
    xt = B.identity(xt)
    if yt is not None:
        yt = B.identity(yt)

    # Sort the targets.
    inv_inds = []
    for i in range(B.shape(xt, 0)):
        inds = np.lexsort(B.to_numpy(xt)[i, ::-1, :])
        inv_inds.append(inv_perm(inds))
        xt[..., i, :, :] = xt[..., i, :, inds]
        if yt is not None:
            yt[..., i, :, :] = yt[..., i, :, inds]

    # Define function which undoes the sorting.

    def unsort(z):
        for i, inds in enumerate(inv_inds):
            z[..., i, :, :] = z[..., i, :, inds]
        return z

    if yt is None:
        return xt, unsort
    else:
        return xt, yt, unsort


def ar_predict(model, xc, yc, xt, pred_num_samples=50, num_samples=5):
    """Autoregressive sampling.

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
        tensor: Marginal mean.
        tensor: Marginal variance.
        tensor: `num_samples` noiseless samples.
    """
    xt, unsort = _sort_targets(xt)

    # Tile to produce multiple samples through batching.
    xc = B.tile(xc[None, ...], pred_num_samples, *((1,) * B.rank(xc)))
    yc = B.tile(yc[None, ...], pred_num_samples, *((1,) * B.rank(yc)))
    xt = B.tile(xt[None, ...], pred_num_samples, *((1,) * B.rank(xt)))

    # Now evaluate the log-likelihood autoregressively.
    samples = []
    for i in range(B.shape(xt, -1)):
        sample = model(xc, yc, xt[..., i : i + 1]).sample()
        samples.append(sample)
        xc = B.concat(xc, xt[..., i : i + 1], axis=-1)
        yc = B.concat(yc, sample, axis=-1)
    samples = B.concat(*samples, axis=-1)

    # Produce noiseless samples by running them through the model once more.
    pred = model(
        xt[:num_samples, ...],
        samples[:num_samples, ...],
        xt[:num_samples, ...],
    )
    noiseless_samples = pred.mean

    mean = B.mean(samples, axis=0)
    var = B.std(samples, axis=0) ** 2

    # Undo the sorting from the beginning.
    mean = unsort(mean)
    var = unsort(var)
    noiseless_samples = unsort(noiseless_samples)

    return mean, var, noiseless_samples


def ar_loglik(state, model, xc, yc, xt, yt, normalise=True):
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

    Returns:
        random state, optional: Random state.
        tensor: Log-likelihoods.
    """
    xt, yt, _ = _sort_targets(xt, yt)

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

import lab as B
import numpy as np

from . import _dispatch
from wbml.util import inv_perm
from .model import Model, _kl
from .disc import Discretisation

__all__ = ["ar_predict", "ar_loglik", "ar_elbo"]


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
            # Generate 64-bytes indices: PyTorch doesn't support indexing with 32 bytes.
            dtype = B.promote_dtypes(B.dtype_int(xt), int)
            state, perm = B.randperm(state, dtype, B.shape(xt, -1))
        else:
            raise RuntimeError(f'Invalid ordering "{order}".')
        inv_perms.append(inv_perm(perm))

        xt[..., i, :, :] = xt[..., i, :, perm]
        if yt is not None:
            yt[..., i, :, :] = yt[..., i, :, perm]

    def unsort(z):
        """Undo the sorting."""
        for i, perm in enumerate(inv_perms):
            z[..., i, :, :] = z[..., i, :, perm]
        return z

    if yt is None:
        return state, xt, unsort
    else:
        return state, xt, yt, unsort


def ar_eval(
    state: B.RandomState,
    model: Model,
    xc,
    yc,
    xt,
    yt=None,
    *,
    num_samples,
):
    def _tile(x):
        return B.tile(x[None, ...], num_samples, *((1,) * B.rank(x)))

    # Tile to produce multiple samples through batching.
    xc = _tile(xc)
    yc = _tile(yc)
    xt = _tile(xt)
    # TODO: Tiling `yt`?

    # Sample and evaluate the model autoregressively.
    preds, samples = [], []
    for i in range(B.shape(xt, -1)):
        state, pred = model(state, xc, yc, xt[..., i : i + 1])
        preds.append(pred)
        xc = B.concat(xc, xt[..., i : i + 1], axis=-1)

        if yt is None:
            state, sample = pred.sample(state)
            samples.append(sample)
            yc = B.concat(yc, sample, axis=-1)
        else:
            yc = B.concat(yc, yt[..., i : i + 1], axis=-1)

    if samples:
        return state, preds, B.concat(*samples, axis=-1)
    else:
        return state, preds


@_dispatch
def ar_predict(
    state: B.RandomState,
    model: Model,
    xc,
    yc,
    xt,
    pred_num_samples=50,
    num_samples=5,
    order="left-to-right",
    grid=True,
    grid_points_per_unit=16,
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
        order (str, optional): Order. Must be one of `"left-to-right"` or `"random"`.
            Defaults to `"left-to-right"`.
        grid (bool, optional): Sample on an intermediate grid. Defaults to `True`.
        grid_points_per_unit (float, optional): Density of the intermediate grid.
            Should be set to the density of the internal discretisation divided by
            four or eight. Defaults to `16`.

    Returns:
        random state, optional: Random state.
        tensor: Marginal mean.
        tensor: Marginal variance.
        tensor: `num_samples` noiseless samples.
    """
    # Determine the intermediate grid, which can just be the targets.
    if grid:
        xg = Discretisation(grid_points_per_unit)(xc, xt)
    else:
        xg = xt

    # Perform sorting.
    state, xg, unsort = _sort_targets(state, xg, order=order)

    # Evaluate the model autoregressively.
    state, preds, yg = ar_eval(
        state,
        model,
        xc,
        yc,
        xg,
        num_samples=pred_num_samples,
    )

    # Produce predictive statistics.
    if grid:
        state, pred = model(state, xg, yg, xt)
        m1s = pred.mean
        m2s = pred.var + m1s**2
        mean = B.mean(m1s, axis=0)
        var = B.mean(m2s, axis=0) - mean**2
    else:
        mean = unsort(B.mean(yg, axis=0))
        var = unsort(B.std(yg, axis=0) ** 2)

    # Produce noiseless samples.
    state, pred = model(
        state,
        xg[:num_samples, ...],
        yg[:num_samples, ...],
        xt,
    )
    noiseless_samples = pred.mean

    return state, mean, var, noiseless_samples


@_dispatch
def ar_predict(model: Model, xc, yc, xt, **kw_args):
    state = B.global_random_state(B.dtype(xt))
    state, mean, var, noiseless_samples = ar_predict(
        state, model, xc, yc, xt, **kw_args
    )
    return mean, var, noiseless_samples


@_dispatch
def ar_elbo(
    state: B.RandomState,
    model: Model,
    xc,
    yc,
    xt,
    yt,
    num_samples=20,
    normalise=True,
    order="left-to-right",
    grid_points_per_unit=16,
):
    """Autoregressive ELBO

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (tensor): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (tensor): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number o samples. Defaults to 1.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `True`.
        order (str, optional): Order. Must be one of `"left-to-right"` or `"random"`.
            Defaults to `"left-to-right"`.
        grid_points_per_unit (float, optional): Density of the intermediate grid.
            Should be set to the density of the internal discretisation divided by
            four or eight. Defaults to `16`.

    Returns:
        random state, optional: Random state.
        tensor: ELBOs.
    """
    xg = Discretisation(grid_points_per_unit)(xc, xt)

    # Perform sorting.
    state, xg, unsort = _sort_targets(state, xg, order=order)

    # Construct proposal and prior.
    state, qs, yg = ar_eval(
        state,
        model,
        B.concat(xc, xt, axis=-1),
        B.concat(yc, yt, axis=-1),
        xg,
        num_samples=num_samples,
    )
    state, ps = ar_eval(state, model, xc, yc, xg, yg, num_samples=num_samples)
    weight = 0
    for i, (q, p) in enumerate(zip(qs, ps)):
        weight = weight + p.logpdf(yg[..., i : i + 1]) - q.logpdf(yg[..., i : i + 1])

    state, pred = model(state, xg, yg, xt)
    elbos = B.logsumexp(pred.logpdf(yt) + weight, axis=0)
    print(elbos)

    # # Compute ELBO.
    # state, pred = model(state, xg, yg, xt)
    # recs = B.mean(pred.logpdf(yt), axis=0)
    # kls = B.mean(sum([_kl(q, p) for q, p in zip(qs, ps)], 0), axis=0)
    # elbos = recs - kls
    # print(elbos)

    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.shape(xt, -1)

    return state, elbos


@_dispatch
def ar_elbo(model: Model, xc, yc, xt, yt, **kw_args):
    state = B.global_random_state(B.dtype(xt))
    state, res = ar_elbo(state, model, xc, yc, xt, yt, **kw_args)
    return res


@_dispatch
def ar_loglik(
    state: B.RandomState,
    model: Model,
    xc,
    yc,
    xt,
    yt,
    normalise=True,
    order="left-to-right",
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
        order (str, optional): Order. Must be one of `"left-to-right"` or `"random"`.
            Defaults to `"left-to-right"`.

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

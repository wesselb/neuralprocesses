from typing import Union

import lab as B
import numpy as np
from plum import Dispatcher
from wbml.util import inv_perm

from .. import _dispatch
from ..aggregate import Aggregate, AggregateInput
from ..datadims import data_dims
from ..mask import Masked
from ..numdata import num_data
from .model import Model
from .util import tile_for_sampling

__all__ = ["ar_predict", "ar_loglik"]


@_dispatch
def _determine_order(
    state: B.RandomState,
    xt: AggregateInput,
    yt: Union[Aggregate, None],
    order: str,
):
    dispatch = Dispatcher()

    # Compute the given ordering. This is what we'll start from.
    pairs = sum(
        [
            [(i_xt, i_out, i_x) for i_x in range(B.shape(xti, -1))]
            for i_xt, (xti, i_out) in enumerate(xt)
        ],
        [],
    )

    if order in {"random", "given"}:
        if order == "random":
            # Randomly permute.
            state, perm = B.randperm(state, B.dtype_int(xt), len(pairs))
            pairs = [pairs[i] for i in perm]

        # For every output, compute the inverse permutation.
        perms = [[] for _ in range(len(xt))]
        for i_xt, i_out, i_x in pairs:
            perms[i_xt].append(i_x)
        inv_perms = [inv_perm(perm) for perm in perms]

        @dispatch
        def unsort(y: B.Numeric) -> Aggregate:
            """Undo the sorting."""
            # Put every output in its bucket.
            buckets = [[] for _ in range(len(xt))]
            for i_y, (i_xt, _, _) in zip(range(B.shape(y, -1)), pairs):
                buckets[i_xt].append(y[..., i_y : i_y + 1])
            # Now sort the buckets.
            buckets = [[bucket[j] for j in p] for bucket, p in zip(buckets, inv_perms)]
            # Concatenate and return.
            return Aggregate(*(B.concat(*bucket, axis=-1) for bucket in buckets))

        return state, xt, yt, pairs, unsort

    elif order == "left-to-right":
        if len(xt) != 1:
            raise ValueError(f"Left-to-right ordering only works for a single output.")

        # Unpack the only output.
        xt, i_out = xt[0]
        if yt is not None:
            yt = yt[0]
        # Copy it, because we'll modify it.
        xt = B.identity(xt)
        if yt is not None:
            yt = B.identity(yt)

        # Sort the targets.
        xt_np = B.to_numpy(xt)  # Need to be NumPy, because we'll use `np.lexsort`.
        perms = [np.lexsort(batch[::-1, :]) for batch in xt_np]
        for i, perm in enumerate(perms):
            xt[i, :, :] = B.take(xt[i, :, :], perm, axis=-1)
            if yt is not None:
                yt[i, :, :] = B.take(yt[i, :, :], perm, axis=-1)

        # Compute the inverse permutations.
        inv_perms = [inv_perm(perm) for perm in perms]

        @dispatch
        def unsort(z: B.Numeric) -> Aggregate:
            """Undo the sorting."""
            z = B.identity(z)  # Make a copy, because we'll modify it.
            for i, perm in enumerate(inv_perms):
                z[..., i, :, :] = B.take(z[..., i, :, :], perm, axis=-1)
            return Aggregate(z)

        # Pack the one output again.
        xt = AggregateInput((xt, i_out))
        yt = Aggregate(yt)

        return state, xt, yt, pairs, unsort

    else:
        raise RuntimeError(f'Invalid ordering "{order}".')


@_dispatch
def ar_predict(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt: AggregateInput,
    num_samples=50,
    order="random",
):
    """Autoregressive sampling.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (:class:`neuralprocesses.aggregrate.AggregateInput`): Inputs of the target
            set. This must be an aggregate of inputs.
        num_samples (int, optional): Number of samples to produce. Defaults to 50.
        order (str, optional): Order. Must be one of `"random"`, `"given"`, or
            `"left-to-right"`. Defaults to `"random"`.

    Returns:
        random state, optional: Random state.
        tensor: Marginal mean.
        tensor: Marginal variance.
        tensor: `num_samples` noiseless samples.
        tensor: `num_samples` noisy samples.
    """
    # Perform sorting.
    state, xt_ordered, _, pairs, unsort = _determine_order(state, xt, None, order)

    # Tile to produce multiple samples through batching.
    contexts = tile_for_sampling(contexts, num_samples)
    xt_ordered = tile_for_sampling(xt_ordered, num_samples)

    # Predict autoregressively. See also :func:`ar_loglik` below.
    contexts = list(contexts)  # Copy it so we can modify it.
    preds, yt = [], []
    for i_xt, i_out, i_x in pairs:
        xti, _ = xt_ordered[i_xt]

        # Make the selection of the particular input.
        xti = xti[..., i_x : i_x + 1]

        # Predict and sample.
        state, pred = model(state, contexts, AggregateInput((xti, i_out)))
        state, yti = pred.sample(state)
        yti = yti[0]  # It is an aggregate with one element.
        preds.append(pred)
        yt.append(yti)

        # Append to the context.
        xci, yci = contexts[i_out]
        contexts[i_out] = (
            B.concat(xci, xti, axis=-1),
            B.concat(yci, yti, axis=-1),
        )
    yt = unsort(B.concat(*yt, axis=-1))

    # Produce predictive statistics. The means and variance will be aggregates with
    # one element.
    m1 = B.mean(B.concat(*(p.mean[0] for p in preds), axis=-1), axis=0)
    m2 = B.mean(B.concat(*(p.var[0] + p.mean[0] ** 2 for p in preds), axis=-1), axis=0)
    mean, var = unsort(m1), unsort(m2 - m1**2)

    # Produce noiseless samples `ft` by passing the noisy samples through the model once
    # more.
    state, pred = model(state, contexts, xt)
    ft = pred.mean

    return state, mean, var, ft, yt


@_dispatch
def ar_predict(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt: B.Numeric,
    **kw_args,
):
    # Run the model forward once to determine the number of outputs.
    # TODO: Is there a better way to do this?
    state, pred = model(state, contexts, xt)
    d = data_dims(xt)
    d_y = B.shape(pred.mean, -(d + 1))

    # Perform AR prediction.
    state, mean, var, ft, yt = ar_predict(
        state,
        model,
        contexts,
        AggregateInput(*((xt, i) for i in range(d_y))),
        **kw_args,
    )

    # Convert the outputs back from `Aggregate`s to a regular tensors.
    mean = B.concat(*mean, axis=-(d + 1))
    var = B.concat(*var, axis=-(d + 1))
    ft = B.concat(*ft, axis=-(d + 1))
    yt = B.concat(*yt, axis=-(d + 1))

    return state, mean, var, ft, yt


@_dispatch
def ar_predict(
    state: B.RandomState,
    model: Model,
    xc: B.Numeric,
    yc: B.Numeric,
    xt: B.Numeric,
    **kw_args,
):
    # Figure out out how many outputs there are.
    d = data_dims(xc)
    d_y = B.shape(yc, -(d + 1))

    def take(y, i):
        """Take the `i`th output."""
        colon = slice(None, None, None)
        return y[(Ellipsis, slice(i, i + 1)) + (colon,) * d]

    return ar_predict(
        state,
        model,
        [(xc, take(yc, i)) for i in range(d_y)],
        xt,
        **kw_args,
    )


@_dispatch
def ar_predict(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-1]))
    res = ar_predict(state, model, *args, **kw_args)
    state, res = res[0], res[1:]
    B.set_global_random_state(state)
    return res


@_dispatch
def _mask_nans(yc: B.Numeric):
    mask = ~B.isnan(yc)
    if B.any(~mask):
        yc = B.where(mask, yc, B.zero(yc))
        return Masked(yc, mask)
    else:
        return yc


@_dispatch
def _mask_nans(yc: Masked):
    return yc


@_dispatch
def _merge_ycs(yc1: B.Numeric, yc2: B.Numeric):
    return B.concat(yc1, yc2, axis=-1)


@_dispatch
def _merge_ycs(yc1: Masked, yc2: B.Numeric):
    with B.on_device(yc2):
        return _merge_ycs(yc1, Masked(yc2, B.ones(yc2)))


@_dispatch
def _merge_ycs(yc1: B.Numeric, yc2: Masked):
    with B.on_device(yc1):
        return _merge_ycs(Masked(yc1, B.ones(yc1)), yc2)


@_dispatch
def _merge_ycs(yc1: Masked, yc2: Masked):
    return Masked(
        _merge_ycs(yc1.y, yc2.y),
        _merge_ycs(yc1.mask, yc2.mask),
    )


@_dispatch
def _merge_contexts(xc1: B.Numeric, yc1, xc2: B.Numeric, yc2):
    xc_merged = B.concat(xc1, xc2, axis=-1)
    yc_merged = _merge_ycs(_mask_nans(yc1), _mask_nans(yc2))
    return xc_merged, yc_merged


@_dispatch
def ar_loglik(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt: AggregateInput,
    yt: Aggregate,
    normalise=False,
    order="random",
):
    """Autoregressive log-likelihood.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (:class:`neuralprocesses.aggregrate.AggregateInput`): Inputs of the target
            set. This must be an aggregate of inputs.
        yt (:class:`neuralprocesses.aggregrate.Aggregate`): Outputs of the target
            set. This must be an aggregate of outputs.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        order (str, optional): Order. Must be one of `"random"`, `"given"`, or
            `"left-to-right"`. Defaults to `"random"`.

    Returns:
        random state, optional: Random state.
        tensor: Log-likelihoods.
    """
    state, xt, yt, pairs, _ = _determine_order(state, xt, yt, order)

    # Below, `i_x` will refer to the index of the inputs, and `i_xt` will refer to the
    # index of `xt`. Note that `i_xt` then does _not_ refer to the index of the output.
    # The index of the output `i_out` will be `i_out = xt[j][0]`.

    # Evaluate the log-likelihood autoregressively.
    contexts = list(contexts)  # Copy it so we can modify it.
    logpdfs = []
    for i_xt, i_out, i_x in pairs:
        xti, _ = xt[i_xt]
        yti = yt[i_xt]

        # Make the selection of the particular input.
        xti = xti[..., i_x : i_x + 1]
        yti = yti[..., i_x : i_x + 1]

        # Compute logpdf.
        state, pred = model(state, contexts, AggregateInput((xti, i_out)))
        logpdfs.append(pred.logpdf(Aggregate(yti)))

        # Append to the context.
        xci, yci = contexts[i_out]
        contexts[i_out] = _merge_contexts(xci, yci, xti, yti)
    logpdf = sum(logpdfs)

    if normalise:
        # Normalise by the number of targets.
        logpdf = logpdf / num_data(xt, yt)

    return state, logpdf


@_dispatch
def ar_loglik(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt: B.Numeric,
    yt: B.Numeric,
    **kw_args,
):
    return ar_loglik(
        state,
        model,
        contexts,
        AggregateInput(*((xt, i) for i in range(B.shape(yt, -2)))),
        Aggregate(*(yt[..., i : i + 1, :] for i in range(B.shape(yt, -2)))),
        **kw_args,
    )


@_dispatch
def ar_loglik(state: B.RandomState, model: Model, xc, yc, xt, yt, **kw_args):
    return ar_loglik(state, model, [(xc, yc)], xt, yt, **kw_args)


@_dispatch
def ar_loglik(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-2]))
    state, res = ar_loglik(state, model, *args, **kw_args)
    B.set_global_random_state(state)
    return res

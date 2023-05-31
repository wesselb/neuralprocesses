import lab as B
import numpy as np

from .. import _dispatch
from ..aggregate import Aggregate, AggregateInput
from ..coding import code, code_track, recode_stochastic
from ..dist import AbstractDistribution
from ..numdata import num_data
from ..parallel import Parallel
from .model import Model
from .util import compress_contexts
from .util import fix_noise as fix_noise_in_pred
from .util import sample

__all__ = ["elbo"]


@_dispatch
def elbo(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt,
    yt,
    *,
    num_samples=1,
    normalise=False,
    subsume_context=False,
    fix_noise=None,
    **kw_args,
):
    """ELBO objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        subsume_context (bool, optional): Subsume the context set into the target set.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.

    Returns:
        random state, optional: Random state.
        tensor: ELBOs.
    """
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)

    if subsume_context:
        # Only here also update the targets.
        contexts_q, xt, yt = _merge_context_target(contexts, xt, yt)
    else:
        contexts_q, _, _ = _merge_context_target(contexts, xt, yt)

    # Construct prior.
    xz, pz, h = code_track(
        model.encoder,
        *compress_contexts(contexts),
        xt,
        root=True,
        dtype_lik=float64,
        **kw_args,
    )

    # Construct posterior.
    qz = recode_stochastic(
        model.encoder,
        pz,
        *compress_contexts(contexts_q),
        h,
        root=True,
        dtype_lik=float64,
        **kw_args,
    )

    # Sample from posterior.
    shape = () if num_samples is None else (num_samples,)
    state, z = sample(state, qz, *shape)
    z = B.cast(float, z)

    # Run sample through decoder.
    _, d = code(
        model.decoder,
        xz,
        z,
        xt,
        dtype_lik=float64,
        root=True,
        **kw_args,
    )
    d = fix_noise_in_pred(d, fix_noise)

    # Compute the ELBO.
    elbos = B.mean(d.logpdf(B.cast(float64, yt)), axis=0) - _kl(qz, pz)

    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.cast(float64, num_data(xt, yt))

    return state, elbos


@_dispatch
def elbo(state: B.RandomState, model: Model, xc, yc, xt, yt, **kw_args):
    return elbo(state, model, [(xc, yc)], xt, yt, **kw_args)


@_dispatch
def elbo(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-1]))
    state, elbos = elbo(state, model, *args, **kw_args)
    B.set_global_random_state(state)
    return elbos


@_dispatch
def _kl(q: AbstractDistribution, p: AbstractDistribution):
    return q.kl(p)


@_dispatch
def _kl(q: Parallel, p: Parallel):
    return sum([_kl(qi, pi) for qi, pi in zip(q, p)])


@_dispatch
def _merge_context_target(contexts: list, xt: B.Numeric, yt: B.Numeric):
    (xc, yc), other_contexts = contexts[0], contexts[1:]

    # Subsume context set into the target set.
    xt = B.concat(xc, xt, axis=-1)
    yt = B.concat(yc, yt, axis=-1)

    # At this point, `(xt, yt)` contains all data. Hence, make this the context set
    # for the approximate posterior.
    contexts_q = [(xt, yt)] + other_contexts

    return contexts_q, xt, yt


@_dispatch
def _merge_context_target(contexts: list, xt: AggregateInput, yt: Aggregate):
    updated_xt, updated_yt = [], []
    q_context_updates = [None] * len(contexts)

    for (xti, i), yti in zip(xt, yt):
        xci, yci = contexts[i]

        # Subsume context set into the target set for output `i`.
        xti = B.concat(xci, xti, axis=-1)
        yti = B.concat(yci, yti, axis=-1)

        # At this point, `(xti, yti)` contains all data for output `i`. Hence, make this
        # the context set for output `i` for the approximate posterior.
        if q_context_updates[i] is not None:
            raise ValueError(
                "Aggregate target inputs specified the same output multiple times."
            )
        q_context_updates[i] = (xti, yti)

        updated_xt.append((xti, i))
        updated_yt.append(yti)

    # Update the target inputs.
    xt = AggregateInput(*updated_xt)
    yt = Aggregate(*updated_yt)

    # Construct the context set for the approximate posterior by applying the recorded
    # updates to `contexts`.
    contexts_q = list(contexts)
    for i, update in enumerate(q_context_updates):
        if update:
            contexts_q[i] = update

    return contexts_q, xt, yt

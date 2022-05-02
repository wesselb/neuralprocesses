import lab as B
import numpy as np

from .model import Model
from .util import sample, fix_noise
from .. import _dispatch
from ..coding import code, code_track, recode_stochastic
from ..dist import AbstractMultiOutputDistribution
from ..parallel import Parallel

__all__ = ["elbo"]


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
    normalise=False,
    subsume_context=False,
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
            Defaults to `False`.
        subsume_context (bool, optional): Subsume the context set into the target set.
            Defaults to `False`.
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
        x_all = xt
        y_all = yt
    else:
        x_all = B.concat(xc, xt, axis=-1)
        y_all = B.concat(yc, yt, axis=-1)

    # Construct prior.
    xz, pz, h = code_track(
        model.encoder,
        xc,
        yc,
        xt,
        root=True,
        dtype_lik=float64,
        **kw_args,
    )

    # Construct posterior.
    qz = recode_stochastic(
        model.encoder,
        pz,
        x_all,
        y_all,
        h,
        root=True,
        dtype_lik=float64,
        **kw_args,
    )

    # Sample from poster.
    state, z = sample(state, qz, num=num_samples)
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
    d = fix_noise(d, epoch)

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
    return sum([_kl(qi, pi) for qi, pi in zip(q, p)])

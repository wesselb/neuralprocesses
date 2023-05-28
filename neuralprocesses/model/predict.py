import lab as B
import numpy as np

from .. import _dispatch
from ..aggregate import Aggregate
from ..dist import shape_batch
from .model import Model

__all__ = ["predict"]


@_dispatch
def predict(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt,
    *,
    num_samples=50,
    batch_size=16,
):
    """Use a model to predict.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        num_samples (int, optional): Number of samples to produce. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 16.

    Returns:
        random state, optional: Random state.
        tensor: Marginal mean.
        tensor: Marginal variance.
        tensor: `num_samples` noiseless samples.
        tensor: `num_samples` noisy samples.
    """
    float = B.dtype_float(xt)
    float64 = B.promote_dtypes(float, np.float64)

    # Collect noiseless samples, noisy samples, first moments, and second moments.
    ft, yt = [], []
    m1s, m2s = [], []

    done_num_samples = 0
    while done_num_samples < num_samples:
        # Limit the number of samples at the batch size.
        this_num_samples = min(num_samples - done_num_samples, batch_size)

        state, pred = model(
            state,
            contexts,
            xt,
            dtype_enc_sample=float,
            # Run likelihood with `float64`s to ease the numerics as much as possible.
            dtype_lik=float64,
            num_samples=this_num_samples,
        )

        # If the number of samples is equal to one but `num_samples > 1`, then the
        # encoding was a `Dirac`, so we can stop batching. In this case, we can
        # efficiently compute everything that we need and exit.
        if this_num_samples > 1 and shape_batch(pred, 0) == 1:
            state, ft = pred.noiseless.sample(state, num_samples)
            state, yt = pred.sample(state, num_samples)
            # If `pred` or `pred.noiseless` were `Dirac`s, then `ft` or `yt` might not
            # have the right number of samples.
            ft = _possibly_tile(ft, num_samples)
            yt = _possibly_tile(yt, num_samples)
            return (
                state,
                # Squeeze the newly introduced sample dimension.
                B.squeeze(pred.mean, axis=0),
                B.squeeze(pred.var, axis=0),
                # Squeeze the previously introduced sample dimension.
                B.squeeze(ft, axis=1),
                B.squeeze(yt, axis=1),
            )

        # Produce samples.
        state, sample = pred.noiseless.sample(state)
        ft.append(sample)
        state, sample = pred.sample(state)
        yt.append(sample)

        # Produce moments.
        m1s.append(pred.mean)
        m2s.append(B.add(pred.var, B.multiply(m1s[-1], m1s[-1])))

        done_num_samples += this_num_samples

    # Stack samples.
    ft = B.concat(*ft, axis=0)
    yt = B.concat(*yt, axis=0)

    # Compute marginal statistics.
    m1 = B.mean(B.concat(*m1s, axis=0), axis=0)
    m2 = B.mean(B.concat(*m2s, axis=0), axis=0)
    mean, var = m1, B.subtract(m2, B.multiply(m1, m1))

    return state, mean, var, ft, yt


@_dispatch
def predict(state: B.RandomState, model: Model, xc, yc, xt, **kw_args):
    return predict(state, model, [(xc, yc)], xt, **kw_args)


@_dispatch
def predict(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-1]))
    res = predict(state, model, *args, **kw_args)
    state, res = res[0], res[1:]
    B.set_global_random_state(state)
    return res


@_dispatch
def _possibly_tile(x: B.Numeric, n: B.Int):
    if B.shape(x, 0) == 1 and n > 1:
        return B.tile(x, n, *((1,) * (B.rank(x) - 1)))
    else:
        return x


@_dispatch
def _possibly_tile(x: Aggregate, n: B.Int):
    return Aggregate(*(_possibly_tile(xi, n) for xi in x))

import lab as B
import numpy as np

from .model import Model
from .. import _dispatch

__all__ = ["predict"]


@_dispatch
def predict(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt,
    *,
    pred_num_samples=50,
    num_samples=5,
):
    """Use a model to predict.

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

    Returns:
        random state, optional: Random state.
        tensor: Marignal mean.
        tensor: Marginal variance.
        tensor: `num_samples` noiseless samples.
    """
    float = B.dtype_float(xt)
    float64 = B.promote_dtypes(float, np.float64)

    # Predict marginal statistics.
    m1s, m2s = [], []
    for _ in range(pred_num_samples):
        state, pred = model(state, contexts, xt)
        m1s.append(pred.mean)
        m2s.append(pred.var + pred.mean**2)
    m1 = B.mean(B.stack(*m1s, axis=0), axis=0)
    m2 = B.mean(B.stack(*m2s, axis=0), axis=0)
    mean, var = m1, m2 - m1**2

    # Produce noiseless samples.
    samples = []
    for _ in range(num_samples):
        state, pred_noiseless = model(
            state,
            contexts,
            xt,
            dtype_enc_sample=float,
            dtype_lik=float64,
            noiseless=True,
        )
        # Try sampling with increasingly higher regularisation.
        epsilon_before = B.epsilon
        while True:
            try:
                state, sample = pred_noiseless.sample(state)
                samples.append(sample)
                break
            except Exception as e:
                B.epsilon *= 10
                if B.epsilon > 1e-3:
                    # Reset regularisation before failing.
                    B.epsilon = epsilon_before
                    raise e
        B.epsilon = epsilon_before  # Reset regularisation after success.
    samples = B.stack(*samples, axis=0)

    return state, mean, var, samples


@_dispatch
def predict(state: B.RandomState, model: Model, xc, yc, xt, **kw_args):
    return predict(state, model, [(xc, yc)], xt, **kw_args)


@_dispatch
def predict(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-1]))
    state, mean, var, samples = predict(state, model, *args, **kw_args)
    B.set_global_random_state(state)
    return mean, var, samples

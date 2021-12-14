import lab as B
import pytest
import tensorflow as tf
import torch
from numpy.testing import assert_allclose

__all__ = ["approx", "nps", "generate_data"]


def approx(a, b, **kw_args):
    assert_allclose(B.to_numpy(a), B.to_numpy(b), **kw_args)


import neuralprocesses.tensorflow as nps_tf
import neuralprocesses.torch as nps_torch

nps_torch.dtype = torch.float32
nps_torch.dtype32 = torch.float32
nps_torch.dtype64 = torch.float64
nps_tf.dtype = tf.float32
nps_tf.dtype32 = tf.float32
nps_tf.dtype64 = tf.float64


@pytest.fixture(params=[nps_torch, nps_tf])
def nps(request):
    nps = request.param
    nps.dtype = nps.dtype32  # Reset data type to `float32`s.
    return nps


def generate_data(nps, batch_size=4, dim_x=1, dim_y=1, n_context=5, n_target=7):
    xc = B.randn(nps.dtype, batch_size, dim_x, n_context)
    yc = B.randn(nps.dtype, batch_size, dim_y, n_context)
    xt = B.randn(nps.dtype, batch_size, dim_x, n_target)
    yt = B.randn(nps.dtype, batch_size, dim_y, n_target)
    return xc, yc, xt, yt

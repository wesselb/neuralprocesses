import lab.torch as B
import numpy.testing
import pytest
import torch

__all__ = ["approx", "context_set", "target_set"]


approx = numpy.testing.assert_allclose


@pytest.fixture()
def context_set():
    batch_size = 2
    n = 15
    x = B.randn(torch.float32, batch_size, n, 1)
    y = B.randn(torch.float32, batch_size, n, 1)
    return x, y


@pytest.fixture()
def target_set():
    batch_size = 2
    n = 10
    x = B.randn(torch.float32, batch_size, n, 1)
    y = B.randn(torch.float32, batch_size, n, 1)
    return x, y

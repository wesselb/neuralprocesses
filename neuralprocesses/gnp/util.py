import numpy as np
import torch.nn as nn
from plum import Dispatcher
import lab as B

__all__ = ["convert_batched_data", "num_params"]

_dispatch = Dispatcher()


def convert_batched_data(x):
    if B.rank(x) == 1:
        return x[None, :, None]
    elif B.rank(x) == 2:
        raise ValueError(
            f"Shape {B.shape(x)} cannot unambiguously be interpreted as batched data."
        )
    elif B.rank(x) == 3:
        return x
    else:
        raise ValueError(f"Shape {B.shape(x)} cannot be interpreted as batched data.")


@_dispatch
def num_params(x: nn.Module):
    return sum([int(np.prod(p.shape)) for p in x.parameters()])

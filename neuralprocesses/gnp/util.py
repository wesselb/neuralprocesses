import numpy as np
import torch.nn as nn
from plum import Dispatcher

__all__ = ["num_params"]

_dispatch = Dispatcher()


@_dispatch
def num_params(x: nn.Module):
    return sum([int(np.prod(p.shape)) for p in x.parameters()])

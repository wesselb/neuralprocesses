import lab as B
import numpy as np

from . import _dispatch

__all__ = [
    "data_dims",
    "batch",
    "compress_batch_dimensions",
    "modules",
    "register_module",
    "models",
    "register_model",
    "is_nonempty",
]


@_dispatch
def data_dims(x: B.Numeric):
    """Check how many data dimensions the encoding corresponding to an input has.

    Args:
        x (input): Input.

    Returns:
        int: Number of data dimensions.
    """
    return 1


@_dispatch
def data_dims(x: tuple):
    return len(x)


def batch(x, other_dims):
    """Get the shape of the batch of a tensor.

    Args:
        x (tensor): Tensor.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tuple[int]: Shape of batch dimensions.
    """
    return B.shape(x)[:-other_dims]


def compress_batch_dimensions(x, other_dims):
    """Compress multiple batch dimensions of a tensor into a single batch dimension.

    Args:
        x (tensor): Tensor to compress.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tensor: `x` with batch dimensions compressed.
        function: Function to undo the compression of the batch dimensions.
    """
    b = batch(x, other_dims)
    if len(b) == 1:
        return x, lambda x: x
    else:

        def uncompress(x_after):
            return B.reshape(x_after, *b, *B.shape(x_after)[1:])

        return B.reshape(x, int(np.prod(b)), *B.shape(x)[len(b) :]), uncompress


modules = []  #: Registered modules.


def register_module(module):
    """Decorator to register a new module."""
    modules.append(module)
    return module


models = []  #: Registered models.


def register_model(model):
    """Decorator to register a new model."""
    models.append(model)
    return model


def is_nonempty(x):
    """Check if a tensor is not empty.

    Args:
        x (tensor): Tensor.

    Returns:
        bool: `True` if `x` is not empty, otherwise `False`.
    """
    return all([i > 0 for i in B.shape(x)])

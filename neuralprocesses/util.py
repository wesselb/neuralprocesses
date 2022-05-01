import lab as B
import numpy as np
from lab.util import resolve_axis

__all__ = [
    "modules",
    "register_module",
    "models",
    "register_model",
    "is_nonempty",
    "batch",
    "compress_batch_dimensions",
    "split",
    "split_channels",
]

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


def split(z, sizes, axis):
    """Split a tensor into multiple tensors.

    Args:
        z (tensor): Tensor to split.
        sizes (iterable[int]): Sizes of the components.
        axis (int): Axis.

    Returns:
        list[tensor]: Components of the split.
    """

    axis = resolve_axis(z, axis)
    index = [slice(None, None, None)] * B.rank(z)

    components = []
    i = 0
    for size in sizes:
        index[axis] = slice(i, i + size, None)
        components.append(z[tuple(index)])
        i += size

    return components


def split_channels(z, sizes, d):
    """Split a tensor at the channels dimension.

    Args:
        z (tensor): Tensor to split.
        sizes (iterable[int]): Sizes of the components.
        d (int): Dimensionality of the inputs.

    Returns:
        list[tensor]: Components of the split.
    """
    return split(z, sizes, -d - 1)

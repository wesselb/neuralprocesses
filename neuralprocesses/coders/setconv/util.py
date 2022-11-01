from functools import wraps

import lab as B
import numpy as np
from lab.util import resolve_axis

from . import _dispatch

__all__ = [
    "is_framework_module",
    "modules",
    "register_module",
    "models",
    "register_model",
    "composite_coders",
    "register_composite_coder",
    "is_composite_coder",
    "wrapped_partial",
    "is_nonempty",
    "batch",
    "compress_batch_dimensions",
    "split",
    "split_dimension",
    "merge_dimensions",
    "select",
    "with_first_last",
]


@_dispatch
def is_framework_module(x):
    """Check if something is a framework module.

    Args:
        x (object): Object to check.

    Returns:
        bool: `True` if `x` is a framework module, else `False`.
    """
    return False


modules = []  #: Registered modules


def register_module(module):
    """Decorator to register a new module."""
    modules.append(module)
    return module


models = []  #: Registered models


def register_model(model):
    """Decorator to register a new model."""
    models.append(model)
    return model


composite_coders = []  #: Composite coders


def register_composite_coder(coder):
    """Decorator to register a composite coder."""
    composite_coders.append(coder)
    return coder


def is_composite_coder(coder):
    """Check if a coder is composite.

    Args:
        coder (coder): Coder.

    Returns:
        bool: Coder is composite.
    """
    return any([isinstance(coder, c) for c in composite_coders])


def wrapped_partial(f, *partial_args, **partial_kw_args):
    """Like :func:`functools.partial`, but preserves the docstring.

    Args:
        f (function): Function to wrap.
        *partial_args: Partial arguments.
        **partial_kw_args: Partial keyword arguments.

    Returns:
        function: Version of `f` with some arguments and keyword arguments already set.
    """

    @wraps(f)
    def wrapped_f(*args, **kw_args):
        return f(*partial_args, *args, **partial_kw_args, **kw_args)

    return wrapped_f


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


def split_dimension(z, axis, sizes):
    """Split a dimension of a tensor into multiple dimensions.

    Args:
        z (tensor): Tensor to split.
        axis (int): Axis to split
        sizes (iterable[int]): Sizes of new dimensions.

    Returns:
        tensor: Reshaped version of `z`.
    """
    shape = B.shape(z)
    # The indexing below will only be correct for positive `axis`, so resolve the index.
    axis = resolve_axis(z, axis)
    return B.reshape(z, *shape[:axis], *sizes, *shape[axis + 1 :])


def merge_dimensions(z, axis, sizes):
    """Merge dimensions of a tensor into one dimension. This operation is the opposite
    of :func:`split_dimension`.

    Args:
        z (tensor): Tensor to merge.
        axis (int): Axis to merge into.
        sizes (iterable[int]): Sizes of dimensions to merge.

    Returns:
        tensor: Reshaped version of `z`.
    """
    shape = B.shape(z)
    # The indexing below will only be correct for positive `axis`, so resolve the index.
    axis = resolve_axis(z, axis)
    return B.reshape(
        z,
        *shape[: axis - len(sizes) + 1],
        np.prod(sizes),
        *shape[axis + 1 :],
    )


def select(z, i, axis):
    """Select a particular index `i` at axis `axis` without squeezing the tensor.

    Args:
        z (tensor): Tensor to select from.
        i (int): Index to select.
        axis (int): Axis to select from.

    Returns:
        tensor: Selection from `z`.
    """
    axis = resolve_axis(z, axis)
    index = [slice(None, None, None) for _ in range(B.rank(z))]
    index[axis] = slice(i, i + 1, None)
    return z[index]


def with_first_last(xs):
    """Return a generator which indicates whether the returned element is the first or
    last.

    Args:
        xs: Generator to wrap.

    Yields:
        bool: Element is first.
        bool: Element is last.
        object: Element.
    """
    state = {"first": True}

    def first():
        if state["first"]:
            state["first"] = False
            return True
        else:
            return False

    prev = None
    have_prev = False

    cur = None
    have_cur = False

    for x in xs:
        cur = x
        have_cur = True

        if not have_prev:
            # We will need a `prev`, but there is no `prev` yet. Take the current one as
            # `prev` and skip to the next iteration.
            prev = cur
            have_prev = True
            continue

        # We currently have available `prev` and `cur`. We will return `prev` and,
        # after the loop has finished, return `cur` as the last one.
        yield first(), False, prev

        prev = cur

    if have_cur:
        yield first(), True, cur

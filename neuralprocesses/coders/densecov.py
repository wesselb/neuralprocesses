import lab as B
import numpy as np

from .. import _dispatch
from ..datadims import data_dims
from ..util import (
    register_module,
    split_dimension,
    batch,
)

__all__ = [
    "ToDenseCovariance",
    "FromDenseCovariance",
    "DenseCovariancePSDTransform",
]


def _reorder_groups(z, groups, order):
    perm = list(range(B.rank(z)))
    # Assume that the groups are specified from the end, so walk through everything in
    # reversed order.
    group_axes = []
    for g in reversed(groups):
        perm, axes = perm[:-g], perm[-g:]
        group_axes.append(axes)
    group_axes = list(reversed(group_axes))
    perm = perm + sum([group_axes[i] for i in order], [])
    return B.transpose(z, perm=perm)


@register_module
class ToDenseCovariance:
    """Shape a regular encoding into a dense covariance encoding."""


@_dispatch
def code(coder: ToDenseCovariance, xz: None, z: B.Numeric, x, **kw_args):
    c, n = B.shape(z, -2, -1)
    if n != 1:
        raise ValueError("Encoding is not global.")
    sqrt_c = int(B.sqrt(c))
    # Only in this case, we also duplicate the inputs!
    return (None, None), B.reshape(z, *B.shape(z)[:-2], sqrt_c, 1, sqrt_c, 1)


@_dispatch
def code(coder: ToDenseCovariance, xz: tuple, z: B.Numeric, x, **kw_args):
    d = data_dims(xz) // 2
    c = B.shape(z, -2 * d - 1)
    sqrt_c = int(B.sqrt(c))
    z = split_dimension(z, -2 * d - 1, (sqrt_c, sqrt_c))

    # The ordering is now `(..., sqrt_c, sqrt_c, *n, *n)` where the length of
    # `(*n,)` is `d`. We want to swap the last `sqrt_c` with the first `*n`.
    z = _reorder_groups(z, (1, 1, d, d), (0, 2, 1, 3))

    return xz, z


@register_module
class FromDenseCovariance:
    """Shape a dense covariance encoding into a regular encoding."""


@_dispatch
def code(coder: FromDenseCovariance, xz, z, x, **kw_args):
    d = data_dims(xz) // 2
    sqrt_c = B.shape(z, -d - 1)

    # The ordering is `(..., sqrt_c, *n, sqrt_c, *n)` where the length of `(*n,)` is
    # `d`. We want to swap the first `*n` with the last `sqrt_c`.
    z = _reorder_groups(z, (1, d, 1, d), (0, 2, 1, 3))

    # Now merge the separate channel dimensions.
    z = B.reshape(z, *batch(z, 2 * d + 2), sqrt_c * sqrt_c, *B.shape(z)[-2 * d :])

    return xz, z


@register_module
class DenseCovariancePSDTransform:
    """Multiply a dense covariance encoding by itself transposed to ensure that it is
    PSD."""


@_dispatch
def code(coder: DenseCovariancePSDTransform, xz, z: B.Numeric, x, **kw_args):
    d = data_dims(xz) // 2

    # Record the original shape so we can transform back at the end.
    orig_shape = B.shape(z)

    # Compute the lengths of the sides of the covariance.
    len1 = np.prod(B.shape(z)[-d - 1 :])
    len2 = np.prod(B.shape(z)[2 * (-d - 1) : -d - 1])

    # Reshape into matrix, perform PD transform, and reshape back.
    z = B.reshape(z, *B.shape(z)[: 2 * (-d - 1)], len1, len2)
    z = B.matmul(z, z, tr_b=True)
    z = z / 100  # Stabilise the initialisation.
    z = B.reshape(z, *orig_shape)

    return xz, z

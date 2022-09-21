import lab as B
import matrix  # noqa

from . import _dispatch
from .aggregate import Aggregate, AggregateInput
from .datadims import data_dims
from .parallel import Parallel
from .util import register_module

__all__ = ["Materialise", "Concatenate", "Sum"]


@register_module
class Concatenate:
    """Materialise an aggregate encoding by concatenating."""


Materialise = Concatenate  #: Alias of `.Concatenate` for backward compatibility.


@_dispatch
def code(coder: Concatenate, xz, z, x, **kw_args):
    return _merge(xz), _repeat_concat(data_dims(xz), z)


@_dispatch
def _merge(z1, z2, z3, *zs):
    z = _merge(z1, z2)
    for zi in (z3,) + zs:
        z = _merge(z, zi)
    return z


@_dispatch
def _merge(zs: Parallel):
    return _merge(*zs)


@_dispatch(precedence=2)
def _merge(z1: Parallel, z2):
    return _merge(_merge(*z1), z2)


@_dispatch(precedence=1)
def _merge(z1, z2: Parallel):
    return _merge(z1, _merge(*z2))


@_dispatch
def _merge(z):
    return z


@_dispatch
def _merge(z1: None, z2: None):
    return None


@_dispatch
def _merge(z1: None, z2):
    return z2


@_dispatch
def _merge(z1, z2: None):
    return z1


@_dispatch
def _merge(z1: B.Numeric, z2: B.Numeric):
    if B.jit_to_numpy(B.mean(B.abs(z1 - z2))) > B.epsilon:
        raise ValueError("Cannot merge inputs.")
    return z1


@_dispatch
def _merge(z1: tuple, z2: tuple):
    return tuple(_merge(z1i, z2i) for z1i, z2i in zip(z1, z2))


@_dispatch
def _merge(z1: AggregateInput, z2: AggregateInput):
    # Merge indices.
    inds1 = tuple(i for _, i in z1)
    inds2 = tuple(i for _, i in z2)
    if inds1 != inds2:
        raise ValueError("Cannot merge aggregate targets.")

    # Merges values and zip indices to them.
    x1 = tuple(x for x, _ in z1)
    x2 = tuple(x for x, _ in z2)
    return AggregateInput(
        *((_merge(x1i, x2i), i) for (x1i, x2i), i in zip(zip(x1, x2), inds1))
    )


@_dispatch
def _repeat_concat(dims, z1, z2, z3, *zs):
    z = _repeat_concat(dims, z1, z2)
    for zi in (z3,) + zs:
        z = _repeat_concat(dims, z, zi)
    return z


@_dispatch
def _repeat_concat(dims, z: Parallel):
    return _repeat_concat(dims, *z)


@_dispatch(precedence=2)
def _repeat_concat(dims, z1: Parallel, z2):
    return _repeat_concat(dims, _repeat_concat(dims, *z1), z2)


@_dispatch(precedence=1)
def _repeat_concat(dims, z1, z2: Parallel):
    return _repeat_concat(dims, z1, _repeat_concat(dims, *z2))


@_dispatch
def _repeat_concat(dims, z):
    return z


@_dispatch
def _repeat_concat(dims: B.Int, z1: B.Numeric, z2: B.Numeric):
    # One of the two may have an sample dimension, but that's the only discrepancy
    # which is allowed.
    rank, rank2 = B.rank(z1), B.rank(z2)
    if rank == rank2:
        pass  # This is fine, of course.
    elif rank + 1 == rank2:
        z1 = B.expand_dims(z1, axis=0)
    elif rank == rank2 + 1:
        z2 = B.expand_dims(z2, axis=0)
    else:
        raise ValueError(f"Cannot concatenate tensors with ranks {rank} and {rank2}.")
    # The ranks of `z1` and `z2` should now be the same. Take the rank of any.
    rank = B.rank(z1)

    # Broadcast the data dimensions and possible sample dimension. There are `1 + dims`
    # many of them, so perform a loop.
    shape1, shape2 = list(B.shape(z1)), list(B.shape(z2))
    for i in [0] + list(range(rank - 1, rank - 1 - dims, -1)):
        shape_n = max(shape1[i], shape2[i])
        # Zeros cannot be broadcasted. Those must be retained.
        if shape1[i] == 0 or shape2[i] == 0:
            shape_n = 0
        shape1[i] = shape_n
        shape2[i] = shape_n
    z1 = B.broadcast_to(z1, *shape1)
    z2 = B.broadcast_to(z2, *shape2)

    # `z1` and `z2` should now be ready for concatenation.
    return B.concat(z1, z2, axis=-1 - dims)


@_dispatch
def _repeat_concat(dims: Aggregate, z1: Aggregate, z2: Aggregate):
    return Aggregate(
        *(_repeat_concat(di, z1i, z2i) for di, z1i, z2i in zip(dims, z1, z2))
    )


@_dispatch
def _repeat_concat(dims: Aggregate, z1: Aggregate, z2):
    return Aggregate(*(_repeat_concat(di, z1i, z2) for di, z1i in zip(dims, z1)))


@_dispatch
def _repeat_concat(dims: Aggregate, z1, z2: Aggregate):
    return Aggregate(*(_repeat_concat(di, z1, z2i) for di, z2i in zip(dims, z2)))


@_dispatch
def _repeat_concat(dims, z1: Aggregate, z2: Aggregate):
    return Aggregate(*(_repeat_concat(dims, z1i, z2i) for z1i, z2i in zip(z1, z2)))


@_dispatch
def _repeat_concat(dims, z1: Aggregate, z2):
    return Aggregate(*(_repeat_concat(dims, z1i, z2) for z1i in z1))


@_dispatch
def _repeat_concat(dims, z1, z2: Aggregate):
    return Aggregate(*(_repeat_concat(dims, z1, z2i) for z2i in z2))


@register_module
class Sum:
    """Materialise an aggregate encoding by summing."""


@_dispatch
def code(coder: Sum, xz, z, x, **kw_args):
    return _merge(xz), _sum(z)


@_dispatch
def _sum(z: B.Numeric):
    return z


@_dispatch
def _sum(zs: Parallel):
    return sum(zs, 0)

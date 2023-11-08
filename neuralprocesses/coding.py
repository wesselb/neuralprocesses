import matrix  # noqa
from plum import isinstance, issubclass

from . import _dispatch
from .dist import AbstractDistribution, Dirac
from .parallel import Parallel
from .util import is_composite_coder

__all__ = [
    "code",
    "code_track",
    "recode",
    "recode_stochastic",
]


@_dispatch
def code(coder, xz, z, x, **kw_args):
    """Perform a coding operation.

    The default behaviour is to apply `coder` to `z` and return `(xz, coder(z))`.

    Args:
        coder (coder): Coder.
        xz (input): Current inputs corresponding to current encoding.
        z (tensor): Current encoding.
        x (input): Desired inputs.

    Returns:
        tuple[input, tensor]: New encoding.
    """
    if any(
        [
            isinstance(coder, s.types[0])
            and issubclass(s.types[0], object)
            and not issubclass(object, s.types[0])
            for s in code.methods
        ]
    ):
        raise RuntimeError(
            f"Dispatched to fallback implementation for `code`, but specialised "
            f"implementation are available. The arguments are "
            f"`({coder}, {xz}, {z}, {x})`."
        )
    return xz, coder(z)


@_dispatch
def code_track(coder, xz, z, x, **kw_args):
    """Perform a coding operation whilst tracking the sequence of target inputs, also
    called the history. This history can be used to perform the coding operation again
    at that sequence of target inputs exactly.

    Args:
        coder (coder): Coder.
        xz (input): Current inputs corresponding to current encoding.
        z (tensor): Current encoding.
        x (input): Desired inputs.

    Returns:
        input: Input of encoding.
        tensor: Encoding.
        list: History.
    """
    return code_track(coder, xz, z, x, [], **kw_args)


@_dispatch
def code_track(coder, xz, z, x, h, **kw_args):
    if is_composite_coder(coder):
        raise RuntimeError(
            f"Dispatched to fallback implementation of `code_track` for "
            f"`{type(coder)}`, but the coder is composite."
        )
    xz, z = code(coder, xz, z, x, **kw_args)
    return xz, z, h + [x]


@_dispatch
def recode(coder, xz, z, h, **kw_args):
    """Perform a coding operation at an earlier recorded sequence of target inputs,
    called the history.

    Args:
        coder (coder): Coder.
        xz (input): Current inputs corresponding to current encoding.
        z (tensor): Current encoding.
        h (list): Target history.

    Returns:
        input: Input of encoding.
        tensor: Encoding.
        list: Remainder of the target history.
    """
    if is_composite_coder(coder):
        raise RuntimeError(
            f"Dispatched to fallback implementation of `recode` for "
            f"`{type(coder)}`, but the coder is composite."
        )
    xz, z = code(coder, xz, z, h[0], **kw_args)
    return xz, z, h[1:]


@_dispatch
def recode_stochastic(coders: Parallel, codings: Parallel, xc, yc, h, **kw_args):
    """In an existing aggregate coding `codings`, recode the codings that are not
    :class:`.dist.Dirac`s for a new context set.

    Args:
        coders (:class:`.parallel.Parallel`): Coders that producing the codings.
        codings (:class:`.parallel.Parallel`): Codings.
        xc (tensor): Inputs of new context set.
        yc (tensor): Outputs of new context set.
        h (list): History.

    Returns:
        :class:`.parallel.Parallel`: Updated coding.
    """
    return Parallel(
        *(
            recode_stochastic(coder, coding, xc, yc, hi, **kw_args)
            for (coder, coding, hi) in zip(coders, codings, h[0])
        )
    )


@_dispatch
def recode_stochastic(coder, coding: Dirac, xc, yc, h, **kw_args):
    # Do not recode `Dirac`s.
    return coding


# If the coding is aggregate, it can still contain `Dirac`s, so we need to be careful.


@_dispatch
def recode_stochastic(coder, coding, xc, yc, h, **kw_args):
    # Do not recode `Dirac`s.
    return _choose(recode(coder, xc, yc, h, **kw_args)[1], coding)


@_dispatch
def _choose(new: Parallel, old: Parallel):
    return Parallel(*(_choose(x, y) for x, y in zip(new, old)))


@_dispatch
def _choose(new: Dirac, old: Dirac):
    # Do not recode `Dirac`s.
    return old


@_dispatch
def _choose(new: AbstractDistribution, old: AbstractDistribution):
    # Do recode other distributions.
    return new

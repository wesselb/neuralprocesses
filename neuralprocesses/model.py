import lab as B
from matrix.util import indent
from plum import List, Tuple, Union

from . import _dispatch
from .augment import AugmentedInput
from .coding import code
from .parallel import Parallel
from .util import register_module

__all__ = ["Model"]


@_dispatch
def _convert_empty_contexts_to_none(xc, yc: B.Numeric):
    if all([B.shape(yc, i) == 0 for i in range(2, B.rank(yc))]):
        return None, None
    else:
        return xc, yc


@_dispatch
def _convert_empty_contexts_to_none(xc: Parallel, yc: Parallel):
    xc, yc = zip(
        *[_convert_empty_contexts_to_none(xci, yci) for xci, yci in zip(xc, yc)]
    )
    return Parallel(*xc), Parallel(*yc)


@register_module
class Model:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    @_dispatch
    def __call__(self, xc, yc, xt, *, num_samples=1, aux_t=None, **kw_args):
        # Perform augmentation of `xt` with auxiliary target information.
        if aux_t is not None:
            xt = AugmentedInput(xt, aux_t)

        xc, yc = _convert_empty_contexts_to_none(xc, yc)
        xz, z = code(self.encoder, xc, yc, xt, **kw_args)
        _, d = code(self.decoder, xz, z, xt, **kw_args)

        return d

    @_dispatch
    def __call__(
        self,
        contexts: List[Tuple[Union[tuple, B.Numeric], B.Numeric]],
        xt,
        **kw_args,
    ):
        return self(
            Parallel(*(c[0] for c in contexts)),
            Parallel(*(c[1] for c in contexts)),
            xt,
            **kw_args,
        )

    def __str__(self):
        return (
            f"Model(\n"
            + indent(str(self.encoder), " " * 4)
            + ",\n"
            + indent(str(self.decoder), " " * 4)
            + "\n)"
        )

    def __repr__(self):
        return (
            f"Model(\n"
            + indent(repr(self.encoder), " " * 4)
            + ",\n"
            + indent(repr(self.decoder), " " * 4)
            + "\n)"
        )

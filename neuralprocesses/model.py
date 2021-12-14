import lab as B
from matrix.util import indent

from .coding import code
from .util import register_module

__all__ = ["Model"]


@register_module
class Model:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def __call__(
        self,
        xc: B.Numeric,
        yc: B.Numeric,
        xt: B.Numeric,
        num_samples=1,
        **kw_args,
    ):
        if B.shape(xc, 2) == 0:
            xc = None
            yc = None
        xz, z = code(self.encoder, xc, yc, xt, **kw_args)
        _, d = code(self.decoder, xz, z, xt, **kw_args)
        return d

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

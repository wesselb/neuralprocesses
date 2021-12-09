import lab as B
import torch

from .coding import code

__all__ = ["Model"]


class Model(torch.nn.Module):
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

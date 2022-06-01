import lab as B

from .data import SyntheticGenerator, new_batch
from ..dist import UniformContinuous

__all__ = ["SinewaveGenerator"]


class SinewaveGenerator(SyntheticGenerator):
    """Sinewave generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.

    Args:

    Attributes:
    """

    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)

    def generate_batch(self):
        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)
            # # Construct the sinewave and add noise.
            x = B.concat(xc, xt, axis=1)
            f = B.sin(B.pi * x.transpose(1, 2))  # period is set to 2.
            # TODO: figure out meaning of self.h, need to draw random mixing
            # matrix sometimes? what does this mean?
            # if self.h is not None:
            #     f = B.matmul(self.h, f)
            y = f + B.sqrt(self.noise) * B.randn(f)
            # # Finalise batch.
            batch = {}
            set_batch(batch, y[:, :, :nc], y[:, :, nc:], transpose=False)
            return batch

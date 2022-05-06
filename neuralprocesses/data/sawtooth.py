import lab as B

from .data import SyntheticGenerator, new_batch
from ..dist import UniformContinuous

__all__ = ["SawtoothGenerator"]


class SawtoothGenerator(SyntheticGenerator):
    """GP generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.

    Args:
        dist_freq (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the frequency. Defaults to the uniform distribution over
            $[3, 5]$.

    Attributes:
        dist_freq (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the frequency.
    """

    def __init__(self, *args, dist_freq=UniformContinuous(3, 5), **kw_args):
        super().__init__(*args, **kw_args)
        self.dist_freq = dist_freq

    def generate_batch(self):
        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)
            x = B.concat(xc, xt, axis=1)

            # Sample a frequency.
            self.state, freq = self.dist_freq.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )

            # Sample a direction.
            self.state, direction = B.randn(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
                B.shape(x, 2),
            )
            norm = B.sqrt(B.sum(direction * direction, axis=2, squeeze=False))
            direction = direction / norm

            # Sample a uniformly distributed (conditional on frequency) offset.
            self.state, sample = B.rand(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
                1,
            )
            offset = sample / freq

            # Construct the sawtooth and add noise.
            f = (freq * (B.matmul(direction, x, tr_b=True) - offset)) % 1
            if self.h is not None:
                f = B.matmul(self.h, f)
            y = f + B.sqrt(self.noise) * B.randn(f)

            # Finalise batch.
            batch = {}
            set_batch(batch, y[:, :, :nc], y[:, :, nc:], transpose=False)
            return batch

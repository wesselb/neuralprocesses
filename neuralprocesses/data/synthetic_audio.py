import lab as B

from .data import SyntheticGenerator, new_batch
from ..dist import UniformContinuous

__all__ = ["SoundlikeGenerator"]


class SoundlikeGenerator(SyntheticGenerator):
    """Soundlike generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.

    Args:
        dist_decay (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the decay tao. Defaults to the uniform distribution over
            $[2, 7]$.

    Attributes:
        dist_decay (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the decay.
    """

    def __init__(
        self,
        *args,
        dist_decay=UniformContinuous(0.1, 0.3),
        dist_period=UniformContinuous(0.75, 1.25),
        dist_w1=UniformContinuous(50, 70),
        dist_w2=UniformContinuous(50, 70),
        **kw_args,
    ):
        super().__init__(*args, **kw_args)
        self.dist_decay = dist_decay
        self.dist_period = dist_period
        self.dist_w1 = dist_w1
        self.dist_w2 = dist_w2

    @staticmethod
    def _wave(x, tao=5, w1=2, w2=3, shift=0, bounds=None):
        x_shift = x - shift
        y1 = B.cos(w1 * x_shift)
        y2 = B.cos(w2 * x_shift)
        y = B.exp(-x_shift / tao) * (y1 + y2)
        if bounds is not None:
            y[x_shift < bounds[0]] = 0  # set negative values to zero
            y[x_shift > bounds[1]] = 0  # set positive values to zero
        return y

    def generate_batch(self):
        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)
            x = B.concat(xc, xt, axis=1)

            bounds = (-2, 2)  # we should get these values from elsewhere
            # Sample the decay
            self.state, D = self.dist_decay.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )
            # Sample the period
            self.state, T = self.dist_period.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )
            # Sample the frequencies
            self.state, w1 = self.dist_w1.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )
            self.state, w2 = self.dist_w2.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )
            dist_phase = UniformContinuous(-1, 1)
            self.state, shift = dist_phase.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )
            shift = shift * T
            z_upper = B.ceil((bounds[1] - shift) / T)
            z_lower = B.floor((bounds[0] - shift) / T)
            t = x.transpose(1, 2)
            s = B.range(z_lower, z_upper)
            waves = B.zeros(t)
            for k in s:
                w = self._wave(t - T * k - shift, tao=D, w1=w1, w2=w2, bounds=(0, T))
                # wave is set to 0 when outside of bounds
                # as defined, t won't be out of bounds.
                w[t < bounds[0]] = 0
                w[t > bounds[1]] = 0
                waves += w
            response = waves
            # TODO: figure out what self.h does
            # if self.h is not None:
            #     f = B.matmul(self.h, f)
            y = response + B.sqrt(self.noise) * B.randn(response)
            # Finalise batch.
            batch = {}
            set_batch(batch, y[:, :, :nc], y[:, :, nc:], transpose=False)
            return batch

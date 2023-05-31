import lab as B
import numpy as np
from lab.shape import Dimension

from neuralprocesses.data import SyntheticGenerator, new_batch

__all__ = ["BiModalGenerator"]


class BiModalGenerator(SyntheticGenerator):
    """Bi-modal distribution generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.
    """

    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)

    def generate_batch(self):
        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)
            x = B.concat(xc, xt, axis=1)

            # Draw a different random phase, amplitude, and period for every task in
            # the batch.
            self.state, rand = B.rand(
                self.state,
                self.float64,
                3,
                self.batch_size,
                1,  # Broadcast over `n`.
                1,  # There is only one input dimension.
            )
            phase = 2 * B.pi * rand[0]
            amplitude = 1 + rand[1]
            period = 1 + rand[2]

            # Construct the noiseless function.
            f = amplitude * B.sin(phase + (2 * B.pi / period) * x)

            # Add noise with variance.
            probs = B.cast(self.float64, np.array([0.5, 0.5]))
            means = B.cast(self.float64, np.array([-0.1, 0.1]))
            variance = 1
            # Randomly choose from `means` with probabilities `probs`.
            self.state, mean = B.choice(self.state, means, self.batch_size, p=probs)
            self.state, randn = B.randn(
                self.state,
                self.float64,
                self.batch_size,
                # `nc` and `nt` are tensors rather than plain integers. Tell dispatch
                # that they can be interpreted as dimensions of a shape.
                Dimension(nc + nt),
                1,
            )
            noise = B.sqrt(variance) * randn + mean[:, None, None]

            # Construct the noisy function.
            y = f + noise

            batch = {}
            set_batch(batch, y[:, :nc], y[:, nc:])
            return batch

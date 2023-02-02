import lab as B
import stheno
import numpy as np

from .data import new_batch
from .gp import GPGenerator
from ..dist import UniformContinuous

__all__ = ["ScaleMixtureGPGenerator"]


class ScaleMixtureGPGenerator(GPGenerator):

    def __init__(
        self,
        *args,
        min_log10_scale=-1.,
        max_log10_scale=1.,
        kernel_type=stheno.EQ,
        **kw_args,
    ):

        self.log10_scale = (min_log10_scale, max_log10_scale)
        self.kernel_type = kernel_type

        super().__init__(*args, **kw_args)

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A batch, which is a dictionary with keys "xc", "yc", "xt", and "yt".
                Also possibly contains the keys "pred_logpdf" and "pred_logpdf_diag".
        """
        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)

            self.state, log10_scale = UniformContinuous(*self.log10_scale).sample(
                self.state,
                xc.dtype,
            )

            scale = 10 ** log10_scale
            kernel = self.kernel_type().stretch(scale)

            # If `self.h` is specified, then we create a multi-output GP. Otherwise, we
            # use a simple regular GP.
            if self.h is None:
                with stheno.Measure() as prior:
                    f = stheno.GP(kernel)
                    # Construct FDDs for the context and target points.
                    fc = f(xc, self.noise)
                    ft = f(xt, self.noise)
            else:
                with stheno.Measure() as prior:
                    # Construct latent processes and initialise output processes.
                    us = [stheno.GP(kernel) for _ in range(self.dim_y_latent)]
                    fs = [0 for _ in range(self.dim_y)]
                    # Perform matrix multiplication.
                    for i in range(self.dim_y):
                        for j in range(self.dim_y_latent):
                            fs[i] = fs[i] + self.h[i, j] * us[j]
                    # Finally, construct the multi-output GP.
                    f = stheno.cross(*fs)
                    # Construct FDDs for the context and target points.
                    fc = f(
                        tuple(fi(xci) for fi, xci in zip(fs, xcs)),
                        self.noise,
                    )
                    ft = f(
                        tuple(fi(xti) for fi, xti in zip(fs, xts)),
                        self.noise,
                    )

            # Sample context and target set.
            self.state, yc, yt = prior.sample(self.state, fc, ft)

            # Make the new batch.
            batch = {}
            set_batch(batch, yc, yt)

            # Store scale used to generate data
            batch["scale"] = scale

            # Compute predictive logpdfs.
            if self.pred_logpdf or self.pred_logpdf_diag:
                post = prior | (fc, yc)
            if self.pred_logpdf:
                batch["pred_logpdf"] = post(ft).logpdf(yt)
            if self.pred_logpdf_diag:
                batch["pred_logpdf_diag"] = post(ft).diagonalise().logpdf(yt)

            # Set epsilon and delta ranges
            if self.sample_dp_params:

                self.state, epsilon = UniformContinuous(*self.dp_epsilon_range).sample(
                    self.state,
                    yc.dtype,
                    yc.shape[0],
                )

                self.state, log10_delta = UniformContinuous(*self.dp_log10_delta_range).sample(
                    self.state,
                    yc.dtype,
                    yc.shape[0]
                )

                batch["epsilon"] = epsilon.detach().cpu()
                batch["delta"] = delta = 10 ** log10_delta.detach().cpu()

            else:

                batch["epsilon"] = None
                batch["delta"] = None

            return batch

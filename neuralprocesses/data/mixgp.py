import lab as B
import stheno

from .gp import GPGenerator, new_batch
from ..dist import UniformDiscrete

__all__ = ["MixtureGPGenerator"]


class MixtureGPGenerator(GPGenerator):
    def __init__(
        self,
        *args,
        mean_diff=0.0,
        **kw_args,
    ):
        super().__init__(*args, **kw_args)

        self.mean_diff = mean_diff

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A batch, which is a dictionary with keys "xc", "yc", "xt", and "yt".
                Also possibly contains the keys "pred_logpdf" and "pred_logpdf_diag".
        """

        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)

            # If `self.h` is specified, then we create a multi-output GP. Otherwise, we
            # use a simple regular GP.
            if self.h is None:
                with stheno.Measure() as prior:
                    f = stheno.GP(self.kernel)
                    # Construct FDDs for the context and target points.
                    fc = f(xc, self.noise)
                    ft = f(xt, self.noise)
            else:
                with stheno.Measure() as prior:
                    # Construct latent processes and initialise output processes.
                    us = [stheno.GP(self.kernel) for _ in range(self.dim_y_latent)]
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

            self.state, i = UniformDiscrete(0, 1).sample(
                self.state,
                self.int64,
                self.batch_size,
            )
            mean = self.mean_diff / 2 - i * self.mean_diff

            yc = yc + mean[:, None, None]
            yt = yt + mean[:, None, None]

            # Make the new batch.
            batch = {}
            set_batch(batch, yc, yt)

            # Compute predictive logpdfs.
            if self.pred_logpdf or self.pred_logpdf_diag:
                post = prior | (fc, yc)
            if self.pred_logpdf:
                batch["pred_logpdf"] = post(ft).logpdf(yt)
            if self.pred_logpdf_diag:
                batch["pred_logpdf_diag"] = post(ft).diagonalise().logpdf(yt)

            return batch

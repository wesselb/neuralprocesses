import abc

import lab as B
import numpy as np
import stheno
from plum import Dispatcher

__all__ = ["MixtureGenerator", "GPGenerator", "SawtoothGenerator"]

_dispatch = Dispatcher()


class DataGenerator(metaclass=abc.ABCMeta):
    """Data generator.

    Attributes:
        batch_size (int): Number of tasks per batch.
        num_batches (int): Number of batches in an epoch.
    """

    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches

    @abc.abstractmethod
    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A batch, which is a dictionary with keys "xc", "yc", "xt", and "yt".
        """

    def epoch(self):
        """Construct a generator for an epoch.

        Returns:
            generator: Generator for an epoch.
        """

        def lazy_gen_batch():
            return self.generate_batch()

        return (lazy_gen_batch() for _ in range(self.num_batches))


class SyntheticGenerator(DataGenerator):
    """Synthetic data generator.

    Args:
        dtype (dtype): Data type to generate.
        noise (float, optional): Observation noise. Defaults to `5e-2`.
        seed (int, optional): Seed. Defaults to 0.
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        x_ranges (tuple[tuple[float, float]...], optional): Ranges of the inputs. Every
            range corresponds to a dimension of the input, which means that the number
            of ranges determine the dimensionality of the input. Defaults to
            `((-2, 2),)`.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to `1`.
        dim_y_latent (int, optional): If `y_dim > 1`, this specifies the number of
            latent processes. Defaults to `y_dim`.
        num_context_points (int or tuple[int, int], optional): A fixed number of context
            points or a lower and upper bound. Defaults to the range `(1, 50)`.
        num_target_points (int or tuple[int, int], optional): A fixed number of target
            points or a lower and upper bound. Defaults to the fixed number `50`.
        device (str, optional): Device on which to generate data. Defaults to `"cpu"`.

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of `dtype` with 64 bits.
        int64 (dtype): Integer version of `dtype` with 64 bits.
        noise (float): Observation noise.
        batch_size (int): Batch size.
        num_tasks (int): Number of tasks to generate per epoch. Is an integer multiple
            of `batch_size`.
        num_batches (int): Number batches in an epoch.
        dim_x (int): Dimensionality of the inputs.
        dim_y (int): Dimensionality of the outputs.
        dim_y_latent (int): If `dim_y > 1`, the number of latent processes.
        h (int): If `dim_y > 1`, the mixing points.
        num_context_points (tuple[int, int]): Lower and upper bound of the number of
            context points.
        num_target_points (tuple[int, int]): Lower and upper bound of the number of
            target points.
        state (random state): Random state.
        device (str): Device.
    """

    def __init__(
        self,
        dtype,
        seed=0,
        noise=0.05**2,
        batch_size=16,
        num_tasks=2**14,
        x_ranges=((-2, 2),),
        dim_y=1,
        dim_y_latent=None,
        num_context_points=(1, 50),
        num_target_points=50,
        device="cpu",
    ):
        self.dtype = dtype

        # Derive the right floating and integral data types from `dtype`.
        self.float64 = B.promote_dtypes(dtype, np.float64)
        self.int64 = B.dtype_int(self.float64)

        self.device = device

        # The random state must be created on the right device.
        with B.on_device(self.device):
            self.state = B.create_random_state(dtype, seed)

        self.noise = noise

        super().__init__(batch_size, num_tasks // batch_size)
        self.num_tasks = num_tasks
        if self.num_batches * batch_size != num_tasks:
            raise ValueError(
                f"Number of tasks {num_tasks} must be a multiple of "
                f"the batch size {batch_size}."
            )

        self.dim_x = len(x_ranges)
        # Construct tensors for the bounds on the input range. These must be `float64`s.
        with B.on_device(self.device):
            lower = B.stack(*(B.cast(self.float64, l) for l, _ in x_ranges))
            upper = B.stack(*(B.cast(self.float64, u) for _, u in x_ranges))
            self.x_ranges = B.to_active_device(lower), B.to_active_device(upper)
        self.dim_y = dim_y
        self.dim_y_latent = dim_y_latent or dim_y

        if self.dim_y > 1:
            # Draw a random mixing matrix.
            self.state, self.h = B.randn(
                self.state,
                self.float64,
                self.dim_y,
                self.dim_y_latent,
            )

        # Ensure that `num_context_points` and `num_target_points` are tuples of lower
        # bounds and upper bounds.
        if not isinstance(num_context_points, tuple):
            num_context_points = (num_context_points, num_context_points)
        if not isinstance(num_target_points, tuple):
            num_target_points = (num_target_points, num_target_points)
        self.num_context_points = num_context_points
        self.num_target_points = num_target_points

    def epoch(self):
        """Construct a generator for an epoch.

        Returns:
            generator: Generator for an epoch.
        """

        def lazy_gen_batch():
            return self.generate_batch()

        return (lazy_gen_batch() for _ in range(self.num_batches))


class MixtureGenerator(DataGenerator):
    """A mixture of data generators.

    Args:
        *gens (:class:`.data.SyntheticGenerator`): Components of the mixture.
        seed (int, optional): Random seed. Defaults to `0`.

    Attributes:
        gens (tuple[:class:`.data.SyntheticGenerator`]): Components of the mixture.
        num_batches (int): Number batches in an epoch.
        batch_size (int): Number of tasks per batch.
        state (random state): Random state.
    """

    @_dispatch
    def __init__(self, *gens: SyntheticGenerator, seed=0):
        for attr in ["batch_size", "num_batches"]:
            if not all(getattr(gen, attr) == getattr(gens[0], attr) for gen in gens):
                raise ValueError(
                    f"Components of the mixture do not have consistent values for "
                    f"attribute `{attr}`."
                )
        super().__init__(gens[0].batch_size, gens[0].num_batches)
        self.gens = gens
        self.state = B.create_random_state(np.float64, seed=seed)

    def generate_batch(self):
        self.state, i = B.randint(self.state, np.int64, upper=len(self.gens))
        return self.gens[i].generate_batch()


def _sample_inputs(gen):
    # Sample number of context and target points.
    lower, upper = gen.num_context_points
    gen.state, num_context_points = B.randint(
        gen.state, gen.int64, lower=lower, upper=upper + 1
    )
    lower, upper = gen.num_target_points
    gen.state, num_target_points = B.randint(
        gen.state, gen.int64, lower=lower, upper=upper + 1
    )

    # Sample inputs.
    gen.state, rand = B.rand(
        gen.state,
        gen.float64,
        gen.batch_size,
        gen.dim_x,
        int(num_context_points + num_target_points),
    )
    lower, upper = gen.x_ranges
    # Make sure the appropriately shape the lower and upper bounds.
    x = lower[None, :, None] + rand * (upper[None, :, None] - lower[None, :, None])

    # Cast `noise` before moving it to the active device, because Python scalars will
    # not be interpreted as tensors and hence will not be moved to the GPU.
    noise = B.to_active_device(B.cast(gen.float64, gen.noise))

    return x, num_context_points, noise


class GPGenerator(SyntheticGenerator):
    """GP generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel of the GP. Defaults to an
            EQ kernel with length scale `0.25`.
        pred_logpdf (bool, optional): Also compute the logpdf of the target set given
            the context set under the true GP. Defaults to `True`.
        pred_logpdf_diag (bool, optional): Also compute the logpdf of the target set
            given the context set under the true diagonalised GP. Defaults to `True`.

    Attributes:
        kernel (:class:`stheno.Kernel`): Kernel of the GP.
        pred_logpdf (bool): Also compute the logpdf of the target set given the context
            set under the true GP.
        pred_logpdf_diag (bool): Also compute the logpdf of the target set given the
            context set under the true diagonalised GP.
    """

    def __init__(
        self,
        *args,
        kernel=stheno.EQ().stretch(0.25),
        pred_logpdf=True,
        pred_logpdf_diag=True,
        **kw_args,
    ):
        self.kernel = kernel
        self.pred_logpdf = pred_logpdf
        self.pred_logpdf_diag = pred_logpdf_diag
        super().__init__(*args, **kw_args)

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A batch, which is a dictionary with keys "xc", "yc", "xt", and "yt".
                Also possibly contains the keys "pred_logpdf" and "pred_logpdf_diag".
        """
        with B.on_device(self.device):
            batch = {}

            # Sample inputs.
            x, num_context_points, noise = _sample_inputs(self)

            # If `self.y_dim > 1`, then we create a multi-output GP. Otherwise, we
            # use a simple regular GP.
            if self.dim_y == 1:
                f = stheno.GP(self.kernel)
            else:
                with stheno.Measure():
                    # Construct latent processes and initialise output processes.
                    xs = [stheno.GP(self.kernel) for _ in range(self.dim_y_latent)]
                    fs = [0 for _ in range(self.dim_y)]
                    # Perform matrix multiplication.
                    for i in range(self.dim_y):
                        for j in range(self.dim_y_latent):
                            fs[i] = fs[i] + self.h[i, j] * xs[j]
                    # Finally, construct the multi-output GP.
                    f = stheno.cross(*fs)

            # Sample context and target set.
            self.state, y = f(B.transpose(x), noise).sample(self.state)
            # Shuffle the dimensions to line up with the convention in the package.
            # Afterwards, when computing logpdfs, we'll have to be careful to reshape
            # things back. Moreover, we need to be super careful when extracting
            # multiple outputs from the sample: reshape to `(self.y_dim, -1)` or to
            # `(-1, self.y_dim)`?
            y = B.reshape(y, self.batch_size, self.dim_y, -1)
            xc = x[:, :, :num_context_points]
            yc = y[:, :, :num_context_points]
            xt = x[:, :, num_context_points:]
            yt = y[:, :, num_context_points:]

            # Compute predictive logpdfs.
            if self.pred_logpdf or self.pred_logpdf_diag:
                # Compute posterior and predictive distribution.
                obs = (f(B.transpose(xc), noise), B.reshape(yc, self.batch_size, -1, 1))
                f_post = f | obs
                fdd = f_post(B.transpose(xt), noise)
                # Prepare `yt` for logpdf computation.
                yt_reshaped = B.reshape(yt, self.batch_size, -1, 1)
            if self.pred_logpdf:
                batch["pred_logpdf"] = fdd.logpdf(yt_reshaped)
            if self.pred_logpdf_diag:
                batch["pred_logpdf_diag"] = fdd.diagonalise().logpdf(yt_reshaped)

            # Convert to the right data type and save.
            batch["xc"] = B.cast(self.dtype, xc)
            batch["yc"] = B.cast(self.dtype, yc)
            batch["xt"] = B.cast(self.dtype, xt)
            batch["yt"] = B.cast(self.dtype, yt)

            return batch


class SawtoothGenerator(SyntheticGenerator):
    """GP generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.

    Args:
        freqs (tuple[float, float], optional): Lower and upper bound of the uniform
            distribution over frequencies. Defaults to `(3, 5)`.

    Attributes:
        freqs (tuple[float, float]): Lower and upper bound of the uniform distribution
            over frequencies.
    """

    def __init__(self, *args, freqs=(3, 5), **kw_args):
        super().__init__(*args, **kw_args)
        self.freqs = freqs

    def generate_batch(self):
        with B.on_device(self.device):
            batch = {}

            # Sample inputs.
            x, num_context_points, noise = _sample_inputs(self)

            # Sample a frequency.
            self.state, rand = B.rand(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
                1,
            )
            lower, upper = self.freqs
            freq = lower + (upper - lower) * rand

            # Sample a direction.
            self.state, direction = B.randn(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
                self.dim_x,
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
            f = (freq * (B.matmul(direction, x) - offset)) % 1
            # Only mix the latent processes if we should generate more than one output.
            if self.dim_y > 1:
                f = B.matmul(self.h, f)
            y = f + B.sqrt(noise) * B.randn(f)

            # Convert to the right data type and save.
            batch["xc"] = B.cast(self.dtype, x[:, :, :num_context_points])
            batch["yc"] = B.cast(self.dtype, y[:, :, :num_context_points])
            batch["xt"] = B.cast(self.dtype, x[:, :, num_context_points:])
            batch["yt"] = B.cast(self.dtype, y[:, :, num_context_points:])

            return batch

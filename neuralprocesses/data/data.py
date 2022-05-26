import abc
import math

import lab as B
import numpy as np
from plum import convert

from ..aggregate import AggregateInput, Aggregate
from ..dist import UniformDiscrete, UniformContinuous, AbstractDistribution
from ..util import split

__all__ = [
    "AbstractGenerator",
    "DataGenerator",
    "SyntheticGenerator",
    "new_batch",
    "apply_task",
]


class AbstractGenerator(metaclass=abc.ABCMeta):
    """Abstract generator.

    Attributes:
        num_batches (int): Number of batches in an epoch.
    """

    @abc.abstractmethod
    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A batch, which is a dictionary with keys "contexts", "xt", and "yt".
        """

    def epoch(self):
        """Construct a generator for an epoch.

        Returns:
            generator: Generator for an epoch.
        """

        def lazy_gen_batch():
            return self.generate_batch()

        return (lazy_gen_batch() for _ in range(self.num_batches))


class DataGenerator(AbstractGenerator):
    """Data generator.

    Args:
        dtype (dtype): Data type.
        seed (int): Seed.
        num_tasks (int): Number of batches in an epoch.
        batch_size (int): Number of tasks per batch.
        device (str): Device.

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of the data type with 64 bytes.
        int64 (dtype): Integral version of the data type with 64 bytes.
        seed (int): Seed.
        batch_size (int): Number of tasks per batch.
        num_batches (int): Number of batches in an epoch.
        device (str): Device.
    """

    def __init__(self, dtype, seed, num_tasks, batch_size, device):
        self.dtype = dtype
        # Derive the right floating and integral data types from `dtype`.
        self.float64 = B.promote_dtypes(dtype, np.float64)
        self.int64 = B.dtype_int(self.float64)

        self.device = device

        # Create the random state on the right device.
        with B.on_device(self.device):
            self.state = B.create_random_state(dtype, seed)

        self.batch_size = batch_size
        self.num_batches = math.ceil(num_tasks / batch_size)


class SyntheticGenerator(DataGenerator):
    """Synthetic data generator.

    Args:
        dtype (dtype): Data type to generate.
        noise (float, optional): Observation noise. Defaults to `5e-2`.
        seed (int, optional): Seed. Defaults to 0.
        seed_params (int, optional): Seed for the model parameters. Defaults to 99.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        batch_size (int, optional): Batch size. Defaults to 16.
        dist_x (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the inputs. Defaults to a uniform distribution over
            $[-2, 2]$.
        dist_x_context (:class:`neuralprocesses.dist.dist.AbstractDistribution`,
            optional): Distribution of the context inputs. Defaults to `dist_x`.
        dist_x_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`,
            optional): Distribution of the target inputs. Defaults to `dist_x`.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to `1`.
        dim_y_latent (int, optional): If `y_dim > 1`, this specifies the number of
            latent processes. Defaults to `y_dim`.
        num_context (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the number of context inputs. Defaults to a uniform
            distribution over $[0, 50]$.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the number of target inputs. Defaults to the fixed number
            50.
        device (str, optional): Device on which to generate data. Defaults to `"cpu"`.

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of `dtype` with 64 bits.
        int64 (dtype): Integer version of `dtype` with 64 bits.
        noise (float): Observation noise.
        num_tasks (int): Number of tasks to generate per epoch. Is an integer multiple
            of `batch_size`.
        batch_size (int): Batch size.
        dist_x_context (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the context inputs.
        dist_x_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the target inputs.
        num_batches (int): Number batches in an epoch.
        num_context (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the number of context inputs.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the number of target inputs.
        dim_y (int): Dimensionality of the outputs.
        dim_y_latent (int): If `dim_y > 1`, the number of latent processes.
        h (int): If `dim_y > 1`, the mixing points.
        state (random state): Random state.
        device (str): Device.
    """

    def __init__(
        self,
        dtype,
        seed=0,
        seed_params=99,
        noise=0.05**2,
        num_tasks=2**14,
        batch_size=16,
        dist_x=UniformContinuous(-2, 2),
        dist_x_context=None,
        dist_x_target=None,
        dim_y=1,
        dim_y_latent=None,
        num_context=UniformDiscrete(0, 50),
        num_target=UniformDiscrete(50, 50),
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size, device)

        self.dist_x_context = convert(dist_x_context or dist_x, AbstractDistribution)
        self.dist_x_target = convert(dist_x_target or dist_x, AbstractDistribution)
        self.num_context = convert(num_context, AbstractDistribution)
        self.num_target = convert(num_target, AbstractDistribution)

        # Use a separate random state for the parameters. This random must be created
        # on the CPU, because the particular random sequence can depend on the device.
        with B.on_device("cpu"):
            state_params = B.create_random_state(dtype, seed_params)

            self.dim_y = dim_y
            self.dim_y_latent = dim_y_latent or dim_y

            if self.dim_y > 1 or self.dim_y_latent > 1:
                # Draw a random mixing matrix.
                state_params, self.h = B.randn(
                    state_params,
                    self.float64,
                    self.dim_y,
                    self.dim_y_latent,
                )
            else:
                self.h = None

        # Move things to the right device.
        with B.on_device(self.device):
            self.noise = B.to_active_device(B.cast(self.float64, noise))
            if self.h is not None:
                self.h = B.to_active_device(self.h)


def new_batch(gen, dim_y, *, fix_x_across_batch=False, batch_size=None):
    """Sample inputs for a new batch. The sampled inputs and assumed outputs
    are all in `(*b, n, c)` format for easier simulation.

    Args:
        gen (:class:`.DataGenerator`): Data generator.
        dim_y (int): Number of outputs.
        fix_x_across_batch (bool): Fix the inputs across the batch. This can help
            with easier simulation. Defaults to `False`.
        batch_size (int, optional): Batch size. Defaults to `gen.batch_size`.

    Returns:
        function: A function which takes in a batch dictionary, `yc`, and `yt`, and
            which appropriately fill the dictionary with the samples. This function
            also accepts a keyword argument `transpose` which you can set to
            `False` if the outputs are already in `(*b, c, n)` format.
        list[tensor]: The context inputs per output.
        tensor: The context inputs for all outputs concatenated.
        int: The number of context inputs.
        list[tensor]: The target inputs per output.
        tensor: The target inputs for all outputs concatenated.
        int: The number of target inputs.
    """
    # Set the default for `batch_size`.
    batch_size = batch_size or gen.batch_size

    def _sample(dist_num, dist_x):
        ns, xs = [], []
        for _ in range(dim_y):
            gen.state, n = dist_num.sample(gen.state, gen.int64)
            if fix_x_across_batch:
                # Set batch dimension to one and then tile.
                gen.state, x = dist_x.sample(
                    gen.state,
                    gen.float64,
                    1,
                    n,
                )
                x = B.tile(x, batch_size, 1, 1)
            else:
                gen.state, x = dist_x.sample(
                    gen.state,
                    gen.float64,
                    batch_size,
                    n,
                )
            ns.append(n)
            xs.append(x)
        return xs, B.concat(*xs, axis=1), ns, sum(ns)

    # For every output, sample the context and inputs.
    xcs, xc, ncs, nc = _sample(gen.num_context, gen.dist_x_context)
    xts, xt, nts, nt = _sample(gen.num_target, gen.dist_x_target)

    def set_batch(batch, yc, yt, transpose=True):
        """Fill a batch dictionary `batch`.

        Args:
            batch (dict): Batch dictionary to fill.
            yc (tensor): Context outputs with shape `(*b, n, c)` with possibly
                `c = 1`.
            yt (tensor): Target outputs with shape `(*b, n, c)` with possibly
                `c = 1`.
            transpose (bool, optional): Set to `False` if `yc` and `yt` already have
                shape `(*b, c, n)`.
        """
        if transpose:
            yc = B.transpose(yc)
            yt = B.transpose(yt)

        # Split up along the data dimension.
        ycs = split(yc, ncs, axis=2)
        yts = split(yt, nts, axis=2)

        # If the right outputs haven't yet been selected, do so.
        ycs = [
            yci[:, i : i + 1, :] if B.shape(yci, 1) > 1 else yci
            for i, yci in enumerate(ycs)
        ]
        yts = [
            yti[:, i : i + 1, :] if B.shape(yti, 1) > 1 else yti
            for i, yti in enumerate(yts)
        ]

        # Convert to the right data type and channels first format, and save. Note
        # that all inputs still have to be transposed. The outputs, however, are
        # already transposed.
        _t = B.transpose
        _c = lambda x: B.cast(gen.dtype, x)
        batch["contexts"] = [(_c(_t(xci)), _c(yci)) for xci, yci in zip(xcs, ycs)]
        if len(xts) > 1:
            # Need to aggregate them together.
            batch["xt"] = AggregateInput(
                *((_c(_t(xti)), i) for i, xti in enumerate(xts))
            )
            batch["yt"] = Aggregate(*(_c(yti) for yti in yts))
        else:
            # No need for aggregation.
            batch["xt"] = _c(_t(xts[0]))
            batch["yt"] = _c(yts[0])

    return set_batch, xcs, xc, nc, xts, xt, nt


def apply_task(state, dtype, int64, mode, xs, ys, num_target, forecast_start):
    """Construct one of three tasks from data.

    Important:
        This assumes that, for every output, the inputs are the same for every element
        in the batch.

    Args:
        state (random state): Random state.
        dtype (dtype): Target dtype.
        int64 (dtype): Integer version of `dtype` with 64 bits.
        mode (str): Task. Must be one of `"interpolation"`, `"forecasting"`,
            `"reconstruction"`, or `"random"`.
        xs (list[tensor]): For every output, the inputs in `(b, 1, n)` form.
        ys (list[tensor]): For every output, the outputs in `(b, 1, n)` form.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the number of target data points, if applicable to the task.
        forecast_start (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the start of the forecasting task.

    Returns:
        random state: Random state.
        dict: Batch with the task applied.
    """
    if mode == "random":
        state, coin = UniformDiscrete(1, 3).sample(state, int64)
        if coin == 1:
            mode = "interpolation"
        elif coin == 2:
            mode = "forecasting"
        else:
            mode = "reconstruction"

    inds_c = []
    inds_t = []

    if mode == "interpolation":
        # For every output, sample a number of context points separately.
        for x in xs:
            state, nt = num_target.sample(state, int64)
            state, perm = B.randperm(state, int64, B.shape(x, -1))
            inds_c.append(perm[nt:])
            inds_t.append(perm[:nt])

    elif mode == "forecasting":
        # Sample a start time and predict all outputs from that point onwards.
        state, start = forecast_start.sample(state, dtype)
        for x in xs:
            inds_c.append(x[0, 0] < start)
            inds_t.append(x[0, 0] >= start)

    elif mode == "reconstruction":
        # Sample a start time and prediction one output from that point onwards. Flip a
        # coin to decide which output to impute.
        state, start = forecast_start.sample(state, dtype)
        state, j = UniformDiscrete(0, len(ys) - 1).sample(state, int64)
        for i, x in enumerate(xs):
            if i == j:
                # Prediction from `start` onwards.
                inds_c.append(x[0, 0] < start)
                inds_t.append(x[0, 0] >= start)
            else:
                # Predict nothing.
                inds = B.range(int64, B.shape(x, -1))
                inds_c.append(inds)
                inds_t.append(inds[:0])

    else:
        raise ValueError(f'Bad mode "{mode}".')

    return state, {
        "contexts": [
            (B.take(x, inds, axis=-1), B.take(y, inds, axis=-1))
            for x, y, inds in zip(xs, ys, inds_c)
        ],
        "xt": AggregateInput(
            *(
                (B.take(x, inds, axis=-1), i)
                for i, (x, inds) in enumerate(zip(xs, inds_t))
            )
        ),
        "yt": Aggregate(*(B.take(y, inds, axis=-1) for y, inds in zip(ys, inds_t))),
    }

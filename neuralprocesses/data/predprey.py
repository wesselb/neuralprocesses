import lab as B
import numpy as np
import wbml.util
from plum import convert
from wbml.data.predprey import load

from .data import DataGenerator, apply_task
from ..dist import AbstractDistribution
from ..dist.uniform import UniformDiscrete, UniformContinuous

__all__ = ["PredPreyGenerator", "PredPreyRealGenerator"]


def _predprey_step(state, x_y, t, dt, *, alpha, beta, delta, gamma, sigma, scale):
    x = x_y[..., 0]
    y = x_y[..., 1]

    state, randn = B.randn(state, B.dtype(x), 2, *B.shape(x))
    dw = B.sqrt(dt) * randn

    deriv_x = x * (alpha - beta * y)
    deriv_y = y * (-delta + gamma * x)
    # Apply an exponent `1 / 6` to emphasise the noise at lower population levels and
    # prevent the populations from dying out.
    x = x + deriv_x * dt + (x ** (1 / 6)) * sigma * dw[0]
    y = y + deriv_y * dt - (y ** (1 / 6)) * sigma * dw[1]

    # Make sure that the populations never become negative. Mathematically, the
    # populations should remain positive. Note that if we were to `max(x, 0)`, then
    # `x` could become zero. We therefore take the absolute value.
    x = B.abs(x)
    y = B.abs(y)

    t = t + dt

    return state, B.stack(x, y, axis=-1), t


def _predprey_rand_params(state, dtype, batch_size=16):
    state, rand = B.rand(state, dtype, 6, batch_size)

    alpha = 0.2 + 0.6 * rand[0]
    beta = 0.04 + 0.04 * rand[1]
    delta = 0.8 + 0.4 * rand[2]
    gamma = 0.04 + 0.04 * rand[3]

    sigma = 0.5 + 9.5 * rand[4]

    scale = 1 + 4 * rand[5]

    return state, {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "gamma": gamma,
        "sigma": sigma,
        "scale": scale,
    }


def _predprey_simulate(state, dtype, t0, t1, dt, t_target, *, batch_size=16):
    state, params = _predprey_rand_params(state, dtype, batch_size=batch_size)

    # Sort the target times for the validity of the loop below.
    perm = B.argsort(t_target)
    inv_perm = wbml.util.inv_perm(perm)
    t_target = B.take(t_target, perm)

    x_y = 5 + 95 * B.rand(dtype, batch_size, 2)
    t, traj = t0, []

    def collect(t_target, remainder=False):
        while B.shape(t_target, 0) > 0 and (t >= t_target[0] or remainder):
            traj.append((t, x_y))
            t_target = t_target[1:]
        return t_target

    # Run the simulation.
    t_target = collect(t_target)
    while t < t1:
        state, x_y, t = _predprey_step(state, x_y, t, dt, **params)
        t_target = collect(t_target)
    t_target = collect(t_target, remainder=True)

    # Concatenate trajectory into a tensor.
    t, traj = zip(*traj)
    t = B.to_active_device(B.cast(dtype, B.stack(*t)))
    traj = B.stack(*traj, axis=-1)

    # Apply a random scale to the trajectory.
    traj = traj * params["scale"][:, None, None]

    # Undo the sorting.
    t = B.take(t, inv_perm)
    traj = B.take(traj, inv_perm, axis=-1)

    return state, t, traj


def _predprey_select_from_traj(t, y, t_target):
    inds = B.sum(B.cast(B.dtype_int(t), t_target[:, None] > t[None, :]), axis=1)
    return B.take(y, inds, axis=-1)


class PredPreyGenerator(DataGenerator):
    """Predator–prey generator with simulated data.

    Args:
        dtype (dtype): Data type to generate.
        seed (int, optional): Seed. Defaults to 0.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        batch_size (int, optional): Batch size. Defaults to 16.
        big_batch_size (int, optional): Size of the big batch. Defaults to 2048.
        dist_x (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the inputs. Defaults to a uniform distribution over
            $[0, 100]$.
        num_data (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the number of data points. Defaults to a uniform
            distribution over $[150, 250]$.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the number of target inputs. Defaults to the fixed number
            100.
        forecast_start (:class:`neuralprocesses.dist.dist.AbstractDistribution`,
            optional): Distribution of the start of the forecasting task. Defaults to
            a uniform distribution over $[25, 75]$.
        mode (str, optional): Mode. Must be one of `"interpolation"`, `"forecasting"`,
            `"reconstruction"`, or `"random"`. Defaults to `"random"`.
        device (str, optional): Device on which to generate data. Defaults to `"cpu"`.

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of `dtype` with 64 bits.
        int64 (dtype): Integer version of `dtype` with 64 bits.
        num_tasks (int): Number of tasks to generate per epoch. Is an integer multiple
            of `batch_size`.
        batch_size (int): Batch size.
        big_batch_size (int): Sizes of the big batch.
        num_batches (int): Number batches in an epoch.
        dist_x (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Distribution
            of the inputs.
        num_data (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the number of data points.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the number of target inputs.
        forecast_start (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the start of the forecasting task.
        mode (str): Mode.
        state (random state): Random state.
        device (str): Device.
    """

    def __init__(
        self,
        dtype,
        seed=0,
        num_tasks=2**14,
        batch_size=16,
        big_batch_size=2048,
        dist_x=UniformContinuous(0, 100),
        num_data=UniformDiscrete(150, 250),
        num_target=UniformDiscrete(100, 100),
        forecast_start=UniformContinuous(25, 75),
        mode="random",
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size, device)

        self.dist_x = convert(dist_x, AbstractDistribution)
        self.num_data = convert(num_data, AbstractDistribution)
        self.num_target = convert(num_target, AbstractDistribution)
        self.forecast_start = convert(forecast_start, AbstractDistribution)
        self.mode = mode

        self.big_batch_size = big_batch_size
        self._big_batch_x = None
        self._big_batch_y = None
        self._big_batch_num_left = 0

    def _get_from_big_batch(self):
        with B.on_device(self.device):
            if self._big_batch_num_left > 0:
                # There is still some available from the big batch. Take that.
                x, y = self._big_batch_x, self._big_batch_y[: self.batch_size]
                self._big_batch_y = self._big_batch_y[self.batch_size :]
                self._big_batch_num_left -= 1
                # Already convert to the target data type so `generate_batch` doesn't
                # have to deal with this.
                return (
                    B.cast(self.dtype, x),
                    B.cast(self.dtype, y),
                )
            else:
                # Use a resolution of 0.05. Also start ten years before 0 to allow the
                # sim to reach steady state.
                x = B.linspace(self.float64, -10, 100, 2200)

                # For computational efficiency, we will not generate one batch, but
                # `multiplier` many batches.
                multiplier = max(self.big_batch_size // self.batch_size, 1)

                # Simulate the equations.
                self.state, x, y = _predprey_simulate(
                    self.state,
                    self.float64,
                    B.min(x),
                    B.max(x),
                    # Use a budget of 5000 steps.
                    (B.max(x) - B.min(x)) / 5000,
                    x,
                    batch_size=multiplier * self.batch_size,
                )

                # Save the big batch and rerun generation to return the first slice.
                self._big_batch_x = x
                self._big_batch_y = y
                self._big_batch_num_left = multiplier
                return self._get_from_big_batch()

    def generate_batch(self):
        with B.on_device(self.device):
            x_all, y_all = self._get_from_big_batch()

            # Sample inputs.
            self.state, n_hare = self.num_data.sample(self.state, self.int64)
            self.state, x_hare = self.dist_x.sample(self.state, self.dtype, n_hare)
            x_hare = B.tile(B.transpose(x_hare)[None, :, :], self.batch_size, 1, 1)
            self.state, n_lynx = self.num_data.sample(self.state, self.int64)
            self.state, x_lynx = self.dist_x.sample(self.state, self.dtype, n_lynx)
            x_lynx = B.tile(B.transpose(x_lynx)[None, :, :], self.batch_size, 1, 1)

            # Sample data.
            y_hare = _predprey_select_from_traj(x_all, y_all[:, 0:1, :], x_hare[0, 0])
            y_lynx = _predprey_select_from_traj(x_all, y_all[:, 1:2, :], x_lynx[0, 0])

            self.state, batch = apply_task(
                self.state,
                self.dtype,
                self.int64,
                self.mode,
                (x_hare, x_lynx),
                (y_hare, y_lynx),
                self.num_target,
                self.forecast_start,
            )

            return batch


class PredPreyRealGenerator(DataGenerator):
    """The real hare–lynx data.

    Args:
        dtype (dtype): Data type to generate.
        seed (int, optional): Seed. Defaults to 0.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^10.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the number of target inputs. Defaults to the fixed number
            100.
        forecast_start (:class:`neuralprocesses.dist.dist.AbstractDistribution`,
            optional): Distribution of the start of the forecasting task. Defaults to
            a uniform distribution over $[25, 75]$.
        mode (str, optional): Mode. Must be one of `"interpolation"`, `"forecasting"`,
            `"reconstruction"`, or `"random"`.
        device (str, optional): Device on which to generate data. Defaults to `"cpu"`.

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of `dtype` with 64 bits.
        int64 (dtype): Integer version of `dtype` with 64 bits.
        num_tasks (int): Number of tasks to generate per epoch. Is an integer multiple
            of `batch_size`.
        batch_size (int): Batch size.
        forecast_start (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the start of the forecasting task.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Distribution of the number of target inputs.
        mode (str): Mode.
        state (random state): Random state.
        device (str): Device.
    """

    def __init__(
        self,
        dtype,
        seed=0,
        num_tasks=2**10,
        num_target=UniformDiscrete(1, 20),
        forecast_start=UniformContinuous(25, 75),
        mode="interpolation",
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size=1, device=device)

        self.num_target = convert(num_target, AbstractDistribution)
        self.forecast_start = convert(forecast_start, AbstractDistribution)
        self.mode = mode

        # Load the data.
        df = load()

        with B.on_device(self.device):
            # Convert the data frame to the framework tensor type.
            self.x = B.cast(self.dtype, np.array(df.index - df.index[0]))
            self.y = B.cast(self.dtype, np.array(df))
            # Move them onto the GPU and make the shapes right.
            self.x = B.to_active_device(self.x)[None, None, :]
            self.y = B.transpose(B.to_active_device(self.y))[None, :, :]

    def generate_batch(self):
        with B.on_device(self.device):
            self.state, batch = apply_task(
                self.state,
                self.dtype,
                self.int64,
                self.mode,
                (self.x, self.x),
                (self.y[:, 0:1, :], self.y[:, 1:2, :]),
                self.num_target,
                self.forecast_start,
            )
            return batch

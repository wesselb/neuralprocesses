import matplotlib.pyplot as plt
from wbml.plot import tweak

import jax.numpy as jnp
import lab.jax as B

from .data import SyntheticGenerator

__all__ = ["PredPreyGenerator"]


def _predprey_step(state, x_y, t, dt, *, alpha, beta, delta, gamma, sigma, sigma_floor):
    x = x_y[..., 0]
    y = x_y[..., 1]

    m = 25

    deriv_x = x * (alpha - beta * y) * (1 - x / m)
    deriv_y = y * (-delta + gamma * x) * (1 - y / m)

    state, randn = B.randn(state, B.dtype(x), 4, *B.shape(x))
    dw = B.sqrt(dt) * randn

    x = x + deriv_x * dt
    x = x + x * (1 - x / m) * sigma * dw[0] + sigma_floor * dw[1]
    y = y + deriv_y * dt
    y = y - y * (1 - y / m) * sigma * dw[2] + sigma_floor * dw[3]

    x = B.maximum(x, B.zero(x))
    y = B.maximum(y, B.zero(y))

    t = t + dt

    return state, B.stack(x, y, axis=-1), t


def _predprey_rand_params(state, dtype, batch_size=16):
    state, rand = B.rand(state, dtype, 5, batch_size)

    alpha = 0.25 + rand[0]
    beta = alpha * (0.2 + 0.2 * rand[1])
    delta = 0.25 + rand[2]
    gamma = delta * (0.2 + 0.2 * rand[3])

    sigma = 0.05 + 0.1 * rand[4]
    sigma_floor = 1

    return state, {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "gamma": gamma,
        "sigma": sigma,
        "sigma_floor": sigma_floor,
    }


def _predprey_simulate(state, dtype, t0, t1, dt, n_out, *, batch_size=16):
    state, params = _predprey_rand_params(state, dtype, batch_size=batch_size)

    x_y = 10 * B.rand(dtype, batch_size, 2)
    t = t0
    t_out = t0
    traj, ts = [x_y], [t]

    while t < t1:
        state, x_y, t = _predprey_step(state, x_y, t, dt, **params)
        if t >= t_out:
            traj.append(x_y)
            ts.append(t)
            t_out += (t1 - t0) / n_out

    t = B.tile(B.stack(*ts)[None, None, :], 16, 1, 1)
    traj = 7 * B.stack(*traj, axis=-1)

    return state, t, traj


class PredPreyGenerator(SyntheticGenerator):
    """Predator–prey generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`.

    For this class, `x_ranges` initialises to `((0, 100),)` and `dim_y` initialises to
    2.
    """

    def __init__(self, *args, x_ranges=((0, 100),), dim_y=2, **kw_args):
        super().__init__(*args, x_ranges=x_ranges, dim_y=dim_y, **kw_args)
        if not (self.dim_x == 1 and self.dim_y == 2):
            raise RuntimeError("`dim_x` must be 1 and `dim_y` must be 2.")
        if not (
            B.all(self.x_ranges_context[i] == self.x_ranges_target[i]) for i in [0, 1]
        ):
            raise RuntimeError("`x_ranges_context` must be equal to `x_ranges_target`.")

    def generate_batch(self):
        with B.on_device(self.device):
            batch = {}

            # Simulate the equations.
            self.state, x, y = _predprey_simulate(
                self.state,
                self.dtype,
                self.x_ranges_context[0][0],
                self.x_ranges_context[1][0],
                7 / 365,
                200,
                batch_size=self.batch_size,
            )

            # Sample numbers of context and target points.
            lower, upper = self.num_context_points
            self.state, num_context_points = B.randint(
                self.state, self.int64, lower=lower, upper=upper + 1
            )
            lower, upper = self.num_target_points
            self.state, num_target_points = B.randint(
                self.state, self.int64, lower=lower, upper=upper + 1
            )

            self.state, perm = B.randperm(self.state, self.int64, 200)
            x = x[..., perm[: num_context_points + num_target_points]]
            y = y[..., perm[: num_context_points + num_target_points]]

            batch["xc"] = x[:, :, :num_context_points]
            batch["yc"] = y[:, :, :num_context_points]
            batch["xt"] = x[:, :, num_context_points:]
            batch["yt"] = y[:, :, num_context_points:]

            return batch

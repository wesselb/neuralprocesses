import lab as B
import wbml.util

from .data import SyntheticGenerator, _sample_all_inputs_noise

__all__ = ["PredPreyGenerator"]


def _predprey_step(state, x_y, t, dt, *, alpha, beta, delta, gamma, sigma):
    x = x_y[..., 0]
    y = x_y[..., 1]

    m = 25

    state, randn = B.randn(state, B.dtype(x), 2, *B.shape(x))
    dw = B.sqrt(dt) * randn

    deriv_x = x * (alpha - beta * y) * (1 - x / m)
    deriv_y = y * (-delta + gamma * x) * (1 - y / m)
    # Apply an exponent 0.2 to emphasise the noise at lower population levels and
    # prevent the populations from dying out.
    x = x + deriv_x * dt + (x**0.2) * (1 - x / m) * sigma * dw[0]
    y = y + deriv_y * dt - (y**0.2) * (1 - y / m) * sigma * dw[1]

    # Make sure that the populations never become negative. Mathematically, the
    # populations should remain positive. Note that if we were to `max(x, 0)`, then
    # `x` could become zero. We therefore take the absolute value.
    x = B.abs(x)
    y = B.abs(y)

    # Cap the populations by `m`. Mathematically, they should never rise above `m`, but
    # numerically they can due to the non-zero step size.
    x = B.minimum(x, m * B.one(x))
    y = B.minimum(y, m * B.one(y))

    t = t + dt

    return state, B.stack(x, y, axis=-1), t


def _predprey_rand_params(state, dtype, batch_size=16):
    state, rand = B.rand(state, dtype, 5, batch_size)

    alpha = 0.25 + rand[0]
    beta = alpha * (0.2 + 0.2 * rand[1])
    delta = 0.25 + rand[2]
    gamma = delta * (0.2 + 0.2 * rand[3])

    sigma = 0.2 + 0.6 * rand[4]

    return state, {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "gamma": gamma,
        "sigma": sigma,
    }


def _predprey_simulate(state, dtype, t0, t1, dt, t_target, *, batch_size=16):
    state, params = _predprey_rand_params(state, dtype, batch_size=batch_size)

    # Sort the target times for the validity of the loop below.
    perm = B.argsort(t_target)
    inv_perm = wbml.util.inv_perm(perm)
    t_target = B.take(t_target, perm)

    # Note the magic constant 10 here.
    x_y = 10 * B.rand(dtype, batch_size, 2)
    t = t0
    traj, ts = [x_y], [t]

    while t < t1:
        state, x_y, t = _predprey_step(state, x_y, t, dt, **params)
        while B.shape(t_target, 0) > 0 and t >= t_target[0]:
            traj.append(x_y)
            ts.append(t)
            t_target = t_target[1:]

    t = B.tile(B.stack(*ts)[None, None, :], batch_size, 1, 1)
    # Note the magic scaling `7 / 100` here.
    traj = B.stack(*traj, axis=-1) * 7 / 100

    # Undo the sorting.
    t = B.take(t, inv_perm, axis=-1)
    traj = B.take(traj, inv_perm, axis=-1)

    # We now apply a random scaling and offset.
    state, offset = B.rand(state, dtype, batch_size, 2, 1)
    state, scale = B.rand(state, dtype, batch_size, 2, 1)
    traj = 0.25 * offset + (0.5 + scale) * traj

    return state, t, traj


class PredPreyGenerator(SyntheticGenerator):
    """Predator–prey generator.

    Further takes in arguments and keyword arguments from the constructor of
    :class:`.data.SyntheticGenerator`. Moreover, also has the attributes of
    :class:`.data.SyntheticGenerator`. However, the defaults for this class are
    different.
    """

    def __init__(
        self,
        *args,
        x_ranges=((0, 100),),
        num_context_points=(50, 100),
        num_target_points=(100, 100),
        dim_y=2,
        **kw_args
    ):
        super().__init__(
            *args,
            x_ranges=x_ranges,
            num_context_points=num_context_points,
            num_target_points=num_target_points,
            dim_y=dim_y,
            **kw_args,
        )
        if not (self.dim_x == 1 and self.dim_y == 2):
            raise RuntimeError("`dim_x` must be 1 and `dim_y` must be 2.")
        self._big_batch = None
        self._big_batch_num_left = 0

    def generate_batch(self):
        # Attempt to return a batch from the big batch.
        if self._big_batch_num_left > 0:
            n = self.batch_size
            batch = {k: v[:n] for k, v in self._big_batch.items()}
            self._big_batch = {k: v[n:] for k, v in self._big_batch.items()}
            self._big_batch_num_left -= 1
            return batch

        with B.on_device(self.device):
            # For computational efficiency, we will not generate one batch, but
            # `multiplier` many batches.
            multiplier = max(1024 // self.batch_size, 1)

            batch = {}

            # Sample inputs
            x, num_context_points, noise = _sample_all_inputs_noise(self)
            # We fix the inputs across all tasks, because that is what the simulator
            # requires.
            x = x[0, 0]

            # Simulate the equations.
            self.state, _, y = _predprey_simulate(
                self.state,
                self.float64,
                min(self.x_ranges_context[0][0], self.x_ranges_target[0][0]),
                max(self.x_ranges_context[1][0], self.x_ranges_target[1][0]),
                7 / 365,
                x,
                batch_size=multiplier * self.batch_size,
            )
            # Now make the inputs of the right size.
            x = B.tile(x[None, None, :], multiplier * self.batch_size, 1, 1)

            batch["xc"] = B.cast(self.dtype, x[:, :, :num_context_points])
            batch["yc"] = B.cast(self.dtype, y[:, :, :num_context_points])
            batch["xt"] = B.cast(self.dtype, x[:, :, num_context_points:])
            batch["yt"] = B.cast(self.dtype, y[:, :, num_context_points:])

            # Save the big batch.
            self._big_batch_num_left = multiplier
            self._big_batch = batch

            # Call the function again to obtain a batch from the big batch.
            return self.generate_batch()

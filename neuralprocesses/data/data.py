import lab.torch as B
import stheno
import torch

__all__ = ["GPGenerator"]


class GPGenerator:
    """GP generator.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel of the GP. Defaults to an
            EQ kernel with length scale `0.25`.
        noise (float, optional): Observation noise. Defaults to `5e-2`.
        seed (int, optional): Seed. Defaults to `0`.
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        x_range (tuple[float, float], optional): Range of the inputs. Defaults to
            [-2, 2].
        num_context_points (int or tuple[int, int], optional): A fixed number of context
            points or a lower and upper bound. Defaults to the range `(1, 50)`.
        num_target_points (int or tuple[int, int], optional): A fixed number of target
            points or a lower and upper bound. Defaults to the fixed number `50`.
        device (str, optional): Device on which to generate data. If no device is given,
            it will try to use the GPU.
    """

    def __init__(
        self,
        kernel=stheno.EQ().stretch(0.25),
        noise=5e-2,
        seed=0,
        batch_size=16,
        num_tasks=2 ** 14,
        x_range=(-2, 2),
        num_context_points=(1, 50),
        num_target_points=50,
        device=None,
    ):
        self.kernel = kernel
        self.noise = noise

        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_batches = num_tasks // batch_size
        if self.num_batches * batch_size != num_tasks:
            raise ValueError(
                f"Number of tasks {num_tasks} must be a multiple of "
                f"the batch size {batch_size}."
            )
        self.x_range = x_range

        # Ensure that `num_context_points` and `num_target_points` are tuples of lower
        # bounds and upper bounds.
        if not isinstance(num_context_points, tuple):
            num_context_points = (num_context_points, num_context_points)
        if not isinstance(num_target_points, tuple):
            num_target_points = (num_target_points, num_target_points)

        self.num_context_points = num_context_points
        self.num_target_points = num_target_points

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # The random state must be created on the right device.
        with B.on_device(self.device):
            self.state = B.create_random_state(torch.float32, seed)

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: A task, which is a dictionary with keys `x_context`, `y_context`,
                `x_target`, and `y_target`.
        """
        # Sample number of context and target points.
        lower, upper = self.num_context_points
        num_context_points = torch.randint(
            lower, upper + 1, (), generator=self.state, device=self.device
        )
        lower, upper = self.num_target_points
        num_target_points = torch.randint(
            lower, upper + 1, (), generator=self.state, device=self.device
        )

        with B.on_device(self.device):
            # Sample context and target set.
            lower, upper = self.x_range
            shape = (self.batch_size, int(num_context_points + num_target_points), 1)
            self.state, rand = B.rand(self.state, torch.float32, *shape)
            x = lower + rand * (upper - lower)
            noise = B.to_active_device(self.noise)
            self.state, y = stheno.GP(self.kernel)(x, noise).sample(self.state)

        return {
            "x_context": x[:, :num_context_points, :],
            "y_context": y[:, :num_context_points, :],
            "x_target": x[:, num_context_points:, :],
            "y_target": y[:, num_context_points:, :],
        }

    def epoch(self):
        """Construct a generator for an epoch.

        Returns:
            generator: Generator for an epoch.
        """

        def lazy_gen_batch():
            return self.generate_batch()

        return (lazy_gen_batch() for _ in range(self.num_batches))

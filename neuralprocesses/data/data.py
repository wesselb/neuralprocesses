import abc

import numpy as np
import stheno
import torch
import lab as B


__all__ = ["GPGenerator"]


class LambdaIterator:
    """Iterator that repeatedly generates elements from a lambda.

    Args:
        generator (function): Function that generates an element.
        num_elements (int): Number of elements to generate.
    """

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0

    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

    def __iter__(self):
        return self


class DataGenerator(metaclass=abc.ABCMeta):
    """Data generator.

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        x_range (tuple[float, float], optional): Range of the inputs. Defaults to
            [-2, 2].
        max_train_points (int, optional): Number of training points. Defaults to 50.
        max_test_points (int, optional): Number of testing points. Defaults to 50.
    """

    def __init__(
        self,
        batch_size=16,
        num_tasks=2 ** 14,
        x_range=(-2, 2),
        max_train_points=50,
        max_test_points=50,
    ):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_batches = num_tasks // batch_size
        if self.num_batches * batch_size != num_tasks:
            raise ValueError(
                f"Number of tasks {num_tasks} must be a multiple of "
                f"the batch size {batch_size}."
            )
        self.x_range = x_range
        self.max_train_points = max_train_points
        self.max_test_points = max_test_points

    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """

    def generate_batch(self, device):
        """Generate a batch.

        Args:
            device (device): Device.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """
        task = {
            "x": [],
            "y": [],
            "x_context": [],
            "y_context": [],
            "x_target": [],
            "y_target": [],
        }

        # Determine number of test and train points.
        num_train_points = np.random.randint(3, self.max_train_points + 1)
        num_test_points = np.random.randint(3, self.max_test_points + 1)
        num_points = num_train_points + num_test_points

        for i in range(self.batch_size):
            # Sample inputs and outputs.
            lower, upper = self.x_range
            x = lower + np.random.rand(num_points) * (upper - lower)
            y = self.sample(x)

            # Determine indices for train and test set.
            inds = np.random.permutation(x.shape[0])
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            # Record to task.
            task["x"].append(sorted(x))
            task["y"].append(y[np.argsort(x)])
            task["x_context"].append(x[inds_train])
            task["y_context"].append(y[inds_train])
            task["x_target"].append(x[inds_test])
            task["y_target"].append(y[inds_test])

        # Stack batch and convert to PyTorch.
        task = {
            k: torch.tensor(
                B.uprank(B.stack(*v, axis=0), rank=3),
                dtype=torch.float32,
                device=device,
            )
            for k, v in task.items()
        }

        return task

    def epoch(self, device):
        """Construct a generator for an epoch.

        Args:
            device (device): Device.

        Returns:
            generator: Generator for an epoch.
        """
        return LambdaIterator(lambda: self.generate_batch(device), self.num_batches)


class GPGenerator(DataGenerator):
    """Generate samples from a GP with a given kernel.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel with length scale `0.25`.
    """

    def __init__(self, kernel=stheno.EQ().stretch(0.25), **kw_args):
        self.gp = stheno.GP(kernel)
        DataGenerator.__init__(self, **kw_args)

    def sample(self, x):
        return B.squeeze(self.gp(x).sample())

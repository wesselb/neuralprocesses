import lab as B
import numpy as np
import torch
from plum import convert

import neuralprocesses.torch as nps
from .data import DataGenerator, apply_task
from .util import cache
from ..dist import AbstractDistribution
from ..dist.uniform import UniformDiscrete, UniformContinuous

__all__ = ["PhoneGenerator"]


class PhoneGenerator(DataGenerator):
    """Phone generator.

    Args:
        dtype (dtype): Data type to generate.
        seed (int, optional): Seed. Defaults to 0.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        batch_size (int, optional): Batch size. Defaults to 16.
        num_target (:class:`neuralprocesses.dist.dist.AbstractDistribution`, optional):
            Distribution of the number of target inputs. Defaults to the uniform
            distribution over $[100, 256]$.
        forecast_start (:class:`neuralprocesses.dist.dist.AbstractDistribution`,
            optional): Distribution of the start of the forecasting task. Defaults to
            a uniform distribution over $[0.5, 0.75]$.
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
        num_batches (int): Number batches in an epoch.
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
        data_path,
        seed=0,
        batch_size=16,
        num_tasks=2 ** 10,
        mode="interpolation",
        num_data=UniformDiscrete(150, 250),  # how to choose these?
        num_target=UniformDiscrete(100, 100),  # how to choose these?
        forecast_start=UniformContinuous(25, 75),
        device="cpu",
        data_task=None,
    ):
        # super().__init__(*args, **kw_args)
        super().__init__(dtype, seed, num_tasks, batch_size, device=device)
        self.data_path = data_path
        self.data_task = data_task

        self.num_data = convert(num_data, AbstractDistribution)

        self.mode = mode
        self.num_target = convert(num_target, AbstractDistribution)
        self.forecast_start = convert(forecast_start, AbstractDistribution)

        self.utterances = self._load_data(self.data_path, self.data_task)
        self._utterances_i = 0


    @staticmethod
    @cache
    def _load_data(data_path, data_task=None):
        # TODO: Load data based on a chosen set of phones, or set of triphones.
        # as defined by data_task.
        # right now this is just using a fixed dataset of "iy".
        return np.load(data_path, allow_pickle=True)

    def generate_batch(self):
        batch_utterances = []
        for _ in range(self.batch_size):
            if self._utterances_i >= len(self.utterances):
                # We've reached the end of the data set. Shuffle and reset the counter
                # to start another cycle.
                self.state, perm = B.randperm(self.state, self.int64, len(self.utterances))
                self.utterances = [self.utterances[i] for i in perm]
                self._utterances_i = 0
            # Get the next utterance.
            batch_utterances.append(self.utterances[self._utterances_i])
            self._utterances_i += 1

        with B.on_device(self.device):
            smallest_utterance_length = min(len(utterance) for utterance in batch_utterances)
            self.state, n_frames = self.num_data.sample(self.state, self.int64)
            n_frames = min((smallest_utterance_length, n_frames.item()))
            self.state, perm = B.randperm(self.state, self.int64, smallest_utterance_length)
            x = perm[:n_frames]
            x = B.tile(B.transpose(x)[None, :, :], self.batch_size, 1, 1)

            ys = []
            for i, utterance in enumerate(batch_utterances):
                u0 = B.cast(torch.float32, utterance)
                y0 = u0[x[i, :, :]]
                ys.append(y0)
            y = np.stack(ys)
            y = B.cast(torch.float32, y)

            # Utterances are off different lengths.
            # Only take up to smallest_utterance_length
            # within a batch to assure that all utterances are of the same length.

            x = B.to_active_device(x)
            y = B.to_active_device(y)

            state = B.create_random_state(np.int64, seed=99)
            self.state, batch = apply_task(
                self.state,
                self.dtype,
                self.int64,
                self.mode,
                (x, ),
                (y[:, 0:1, :], ),
                self.num_target,
                self.forecast_start,
            )

            return batch

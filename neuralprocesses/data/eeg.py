import numpy as np
from lab import B
from plum import convert
from wbml.data.eeg import load_full as load_eeg

from .data import DataGenerator, apply_task
from .util import cache
from ..dist import AbstractDistribution, UniformContinuous, UniformDiscrete

__all__ = ["EEGGenerator"]

_eeg_all_subjects = [
    337,
    338,
    339,
    340,
    341,
    342,
    344,
    345,
    346,
    347,
    348,
    351,
    352,
    354,
    355,
    356,
    357,
    359,
    362,
    363,
    364,
    365,
    367,
    368,
    369,
    370,
    371,
    372,
    373,
    374,
    375,
    377,
    378,
    379,
    380,
    381,
    382,
    383,
    384,
    385,
    386,
    387,
    388,
    389,
    390,
    391,
    392,
    393,
    394,
    395,
    396,
    397,
    398,
    400,
    402,
    403,
    404,
    405,
    406,
    407,
    409,
    410,
    411,
    412,
    414,
    415,
    416,
    417,
    418,
    419,
    421,
    422,
    423,
    424,
    425,
    426,
    427,
    428,
    429,
    430,
    432,
    433,
    434,
    435,
    436,
    437,
    438,
    439,
    440,
    443,
    444,
    445,
    447,
    448,
    450,
    451,
    453,
    454,
    455,
    456,
    457,
    458,
    459,
    460,
    461,
    1000367,
]


class EEGGenerator(DataGenerator):
    """EEG Generator.

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
        subset (str, optional): Subset of the data. Must be one of `"train"`, `"cv"`, or
            `"eval"`. Defaults to `"train"`.
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

    _data_cache = {}

    def __init__(
        self,
        dtype,
        seed=0,
        num_tasks=2**14,
        batch_size=16,
        num_target=UniformDiscrete(50, 256),  # UniformDiscrete(50, 200),
        forecast_start=UniformContinuous(0.5, 0.75),
        subset="train",
        mode="random",
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size, device)

        self.mode = mode
        self.num_target = convert(num_target, AbstractDistribution)
        self.forecast_start = convert(forecast_start, AbstractDistribution)

        # Load the trails. Use caching to speed up future constructions of the
        # generator.
        self.trials = self._load_trials(subset)
        self._trials_i = 0  # Maintain a counter of how far into the data we are.

    @staticmethod
    @cache
    def _load_trials(subset):
        # Shuffle subjects. Do not modify the below seed!
        state = B.create_random_state(np.float32, seed=99)
        state, perm = B.randperm(state, np.int64, len(_eeg_all_subjects))
        all_subjects = np.array(_eeg_all_subjects)[perm]

        # Split into training, cross-validation, and evaluation data.
        if subset == "train":
            subjects = all_subjects[20:]
        elif subset == "cv":
            subjects = all_subjects[10:20]
        elif subset == "eval":
            subjects = all_subjects[:10]
        else:
            raise ValueError(f'Unknown subset "{subset}" for EEG data.')

        # Load EEG data and construct list of trials.
        data = load_eeg()
        trials = []
        for subject in subjects:
            for n in sorted(data[subject]["trials"].keys()):
                # Select the `n`th trial for the subject.
                trial = data[subject]["trials"][n]["df"]
                trial = trial.loc[:, ["FZ", "F1", "F2", "F3", "F4", "F5", "F6"]]
                if np.abs(np.array(trial)).sum() == 0:
                    # There is no data here! Skip it.
                    continue
                trials.append(trial)
        return trials

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: "contexts", "xt", "yt"
        """
        batch_trials = []
        for _ in range(self.batch_size):
            if self._trials_i >= len(self.trials):
                # We've reached the end of the data set. Shuffle and reset the counter
                # to start another cycle.
                self.state, perm = B.randperm(self.state, self.int64, len(self.trials))
                self.trials = [self.trials[i] for i in perm]
                self._trials_i = 0
            # Get the next trial.
            batch_trials.append(self.trials[self._trials_i])
            self._trials_i += 1

        with B.on_device(self.device):
            # Concatenate the elements of the batch.
            x = np.array(batch_trials[0].index)[None, None, :]
            x = B.tile(x, self.batch_size, 1, 1)
            y = B.stack(*[np.array(yi).T for yi in batch_trials], axis=0)
            # Convert to the right data type and move to right device.
            x = B.to_active_device(B.cast(self.dtype, x))
            y = B.to_active_device(B.cast(self.dtype, y))

            self.state, batch = apply_task(
                self.state,
                self.dtype,
                self.int64,
                self.mode,
                [x for _ in range(B.shape(y, 1))],
                [y[:, i : i + 1, :] for i in range(B.shape(y, 1))],
                self.num_target,
                self.forecast_start,
            )
            return batch

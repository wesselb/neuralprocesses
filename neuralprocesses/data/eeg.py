from lab import B
from wbml.data.eeg import load_full as load_eeg

from ..aggregate import Aggregate, AggregateTargets

import numpy as np

from .data import DataGenerator
from ..dist import UniformDiscrete

__all__ = ["EEGGenerator"]


class EEGGenerator(DataGenerator):
    """EEG Generator.

    Attributes:
        batch_size (int): Number of tasks per batch.
        num_batches (int): Number of batches in an epoch.
    """

    def __init__(
        self,
        split,
        dtype,
        split_seed,
        shuffle_seed,
        num_tasks,
        batch_size,
        device,
        num_targets=UniformDiscrete(1, 256),
    ):

        super().__init__(
            dtype=dtype,
            seed=shuffle_seed,
            num_tasks=num_tasks,
            batch_size=batch_size,
            device=device,
        )

        self.num_targets = num_targets

        all_subjects = [
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

        # Create PRNG states for train/valid/test split and shuffling
        self.split_state = B.create_random_state(np.float32, split_seed)
        self.shuffle_state = B.create_random_state(np.float32, shuffle_seed)

        # Shuffle subjects
        self.split_state, idx = B.randperm(self.split_state, int, len(all_subjects))
        all_subjects = np.array(all_subjects)[idx]
        all_subjects = list(all_subjects)

        # Split into training validation and test data
        if split == "train":
            subjects = all_subjects[20:]

        elif split == "valid":
            subjects = all_subjects[10:20]

        elif split == "test":
            subjects = all_subjects[:10]

        else:
            raise ValueError(f'Unknown split "{split}" for EEG data.')

        # Load EEG data and set list of trials
        data = load_eeg()
        self.trials = self.make_trials(data=data, subjects=subjects)

        # Set counter for resetting at the end of an epoch
        self.i = 0

    def make_trials(self, data, subjects):

        trials = []

        for subject in subjects:
            for n in sorted(data[subject]["trials"].keys()):

                # Select the nth trial for the subject
                trial = data[subject]["trials"][n]["df"]
                trial = trial.reindex(sorted(trial.columns), axis=1)
                trial = trial.loc[:, ["F3", "F4", "F5", "F6", "FZ", "F1", "F2"]]

                # Append the trial to the master list
                trial = trial.iloc[:, :7]
                trials.append(trial)

        return trials

    def generate_batch(self):
        """Generate a batch.

        Returns:
            dict: "contexts", "xt", "yt"
        """

        batch_trials = []

        for _ in range(self.batch_size):
            if self.i >= len(self.trials):

                # Shuffle and cycle
                self.shuffle_state.shuffle(self.trials)
                self.i = 0

            # Get trial
            batch_trials.append(self.trials[self.i])
            self.i += 1

        x = np.repeat(
            np.array(batch_trials[0].index)[None, None, :],
            self.batch_size,
            axis=0,
        )

        # Carefully order the outputs
        y = np.transpose(np.stack(batch_trials, axis=0), (0, 2, 1))
        print(x.shape, y.shape)

        contexts = [(x, y[:, i : i + 1, :]) for i in range(7)]

        n = self.shuffle_state.randint(low=0, high=6)

        contexts = []
        xt = []
        yt = []

        for i in range(7):

            ctx = (x, y[:, i : i + 1, :])

            if i == n:
                self.shuffle_state, k = self.num_targets.sample(self.shuffle_state, int)
                idx = self.shuffle_state.permutation(256)

                c_idx = idx[:k]
                t_idx = idx[k:]

                contexts.append((x[:, :, c_idx], y[:, i : i + 1, c_idx]))

                xt.append(x[:, :, t_idx])
                yt.append(y[:, i : i + 1, t_idx])

            else:
                contexts.append((x[:, :, :], y[:, i : i + 1, :]))

        with B.on_device(self.device):
            contexts = [
                (
                    B.to_active_device(B.cast(self.dtype, x)),
                    B.to_active_device(B.cast(self.dtype, y)),
                )
                for x, y in contexts
            ]

            xt = AggregateTargets(
                *[
                    (B.to_active_device(B.cast(self.dtype, _xt)), i)
                    for i, _xt in enumerate(xt)
                ]
            )

            yt = Aggregate(*[B.to_active_device(B.cast(self.dtype, _yt)) for _yt in yt])

        batch = {
            "contexts": contexts,
            "xt": xt,
            "yt": yt,
        }

        return batch

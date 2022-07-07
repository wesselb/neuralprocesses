from pathlib import Path
from typing import List

import lab as B
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from plum import convert
from pydub import AudioSegment
from tqdm import tqdm

from .data import DataGenerator, apply_task
from .util import cache
from ..dist import AbstractDistribution
from ..dist.uniform import UniformDiscrete, UniformContinuous

__all__ = ["PhoneGenerator"]


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


class _PhoneData:
    def __init__(self, data_path, data_task, train_split=0.70, seed=0):
        # TODO: add error if not in list of all phones
        # TODO: add ability to use all vowels
        # TODO: adapt to words somehow?
        # if data_task not in all_phones:
        #     raise ValueError(f"`data_task` must be one of {all_phones}")
        df = load_phone_df(data_path, data_task)
        df = df.apply(get_signal_data, timit_loc=data_path, axis=1)
        cv_split = 1 - (1 - train_split) / 2
        splits = [int(train_split * len(df)), int(cv_split * len(df))]
        train_df, cv_df, eval_df = np.split(
            df.sample(frac=1, random_state=seed), splits
        )

        self.df = df
        self.train_phones = train_df["phn_data"].values
        self.cv_phones = cv_df["phn_data"].values
        self.eval_phones = eval_df["phn_data"].values


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
        seed=0,
        batch_size=16,
        num_tasks=2**10,
        mode="interpolation",
        num_data=UniformDiscrete(20, 300),  # how to choose these?
        num_target=UniformDiscrete(50, 200),  # how to choose these?
        forecast_start=UniformDiscrete(50, 300),
        device="cpu",
        subset="train",
        data_path="data/timit/",
        data_task=("iy"),  # add default or make default behaviour to load all phones?
    ):
        super().__init__(dtype, seed, num_tasks, batch_size, device)

        if (not isinstance(data_task, tuple)) and (data_task is not None):
            data_task = tuple(data_task)
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        super().__init__(dtype, seed, num_tasks, batch_size, device=device)
        self.data_path = data_path
        self.data_task = data_task

        self.num_data = convert(num_data, AbstractDistribution)

        self.mode = mode
        self.num_target = convert(num_target, AbstractDistribution)
        self.forecast_start = convert(forecast_start, AbstractDistribution)

        # self.utterances = self._load_data(self.data_path, self.data_task)
        data = PhoneGenerator._load_data(self.data_path, self.data_task)
        self.subset = subset
        # data = self._load_data(self.data_path, self.data_task)
        if subset == "train":
            self.utterances = data.train_phones
        elif subset == "cv":
            self.utterances = data.cv_phones
        elif subset == "eval":
            self.utterances = data.eval_phones
        else:
            raise ValueError(f"`subset` must be one of ['train', 'cv', 'eval']")
        self._utterances_i = 0

    @staticmethod
    @cache
    def _load_data(data_path, data_task=("iy")):
        data = _PhoneData(data_path, data_task)
        return data

    def generate_batch(self):
        batch_utterances = []
        for _ in range(self.batch_size):
            if self._utterances_i >= len(self.utterances):
                # We've reached the end of the data set. Shuffle and reset the counter
                # to start another cycle.
                self.state, perm = B.randperm(
                    self.state, self.int64, len(self.utterances)
                )
                self.utterances = [self.utterances[i] for i in perm]
                self._utterances_i = 0
            # Get the next utterance.
            batch_utterances.append(self.utterances[self._utterances_i])
            self._utterances_i += 1

        with B.on_device(self.device):
            smallest_utterance_length = min(
                len(utterance) for utterance in batch_utterances
            )
            self.state, n_frames = self.num_data.sample(self.state, self.int64)
            n_frames = min((smallest_utterance_length, n_frames.item()))
            self.state, perm = B.randperm(
                self.state, self.int64, smallest_utterance_length
            )
            x = perm[:n_frames]
            x = B.tile(B.transpose(x)[None, :, :], self.batch_size, 1, 1)
            x = B.cast(self.dtype, x)

            y = B.zeros(x)
            for i, utterance in enumerate(batch_utterances):
                u0 = B.cast(self.dtype, utterance)
                y0 = u0[x[i, :, :].long()]
                y[i, :, :] = y0

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
                (x,),
                (y[:, 0:1, :],),
                self.num_target,
                self.forecast_start,
            )

            return batch


def get_speaker_dirs(timit_loc: Path):
    speaker_dirs = []
    for f in timit_loc.iterdir():
        if f.is_dir():
            speaker_dirs.append(f)
    return speaker_dirs


def get_sentence_ids(speaker_dir):
    sentence_ids = []
    for f in speaker_dir.iterdir():
        if f.stem not in sentence_ids:
            sentence_ids.append(f.stem)
    return sentence_ids


def load_phones(phn_loc: Path):
    with open(phn_loc, "r") as f:
        phn_lines = f.readlines()
    phn_lines = [l.strip().split() for l in phn_lines]
    phn_df = pd.DataFrame(phn_lines, columns=["start", "end", "phone"])
    phn_df["start"] = phn_df["start"].astype(int)
    phn_df["end"] = phn_df["end"].astype(int)
    return phn_df


def get_signal_data(
    phn_row: pd.Series, timit_loc: Path, full_wav=False, normalize_gain=-20
):
    phn_row = phn_row.copy()
    subdir = timit_loc / f"{phn_row['dialect']}-{phn_row['speaker']}"
    wav_loc = subdir / f"{phn_row['sentence']}.wav"
    aseg = AudioSegment.from_file(wav_loc)
    if normalize_gain is not None:
        aseg = match_target_amplitude(aseg, normalize_gain)
    wav_data = np.array(aseg.get_array_of_samples())
    if full_wav is True:
        phn_data = wav_data
    else:
        start = phn_row["start"]
        end = phn_row["end"]
        phn_data = wav_data[start:end]
    phn_row["phn_data"] = phn_data
    phn_row["fs"] = aseg.frame_rate
    return phn_row


def load_phone_df(timit_loc: Path, phn0: List[str]):
    # Load list of all phones and throw and error if phn0 has values not in list
    speaker_dirs = get_speaker_dirs(timit_loc)
    phn0_dfs = []
    pbar = tqdm(speaker_dirs)
    for speaker_dir in pbar:
        pbar.set_description(speaker_dir.name)
        sentence_ids = get_sentence_ids(speaker_dir)
        for loc in sentence_ids:
            phn_loc = speaker_dir / f"{loc}.phn"
            phn_df = load_phones(phn_loc)
            phn0_df0 = phn_df[np.isin(phn_df["phone"], phn0)].copy()
            phn0_df0["dialect"], phn0_df0["speaker"] = speaker_dir.stem.split("-")
            phn0_df0["sentence"] = loc
            phn0_dfs.append(phn0_df0)
    phn0_df = pd.concat(phn0_dfs)
    phn0_df = phn0_df.reset_index(drop=True)
    return phn0_df


def plot_signal(phn_data, fs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(x=range(len(phn_data)), y=phn_data, ax=ax1)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Frame")

    ax2.specgram(phn_data, Fs=fs)
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time (s)")
    return fig

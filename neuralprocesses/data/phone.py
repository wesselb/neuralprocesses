from pathlib import Path
from typing import List
import librosa

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
tqdm.pandas()


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def librosa_resample(samples, orig_fr, target_fr=22050):
    arr = np.array(samples).astype(np.float32) / 32768  # 16 bit
    arr = librosa.core.resample(
        arr, orig_sr=orig_fr, target_sr=target_fr, res_type="kaiser_best"
    )
    return arr


def get_train_cv_eval_dfs(data_path, data_task, seed=0, frac=1):
    df = load_phone_df(data_path, data_task, frac=frac)
    df["wav_loc"] = df.apply(get_wav_loc, timit_loc=data_path, axis=1)
    phn0_df = df.progress_apply(get_signal_data, axis=1)

    train_df = phn0_df[phn0_df["dataset"] == "TRAIN"]
    test_df = phn0_df[phn0_df["dataset"] == "TEST"]

    # This information is duplicated in dist_x_target
    snippet_start = 0  # put this in some config somewhere?
    snippet_len = 800
    train_seg_df = get_phone_segs(train_df, snippet_start, snippet_len)
    test_seg_df = get_phone_segs(test_df, snippet_start, snippet_len)

    orig_fr = train_df["fs"].iloc[0]  # assume all the same
    # target_fr = 22050
    target_fr = orig_fr
    train_seg_df = train_seg_df.apply(
        librosa_resample, orig_fr=orig_fr, target_fr=target_fr
    )
    test_seg_df = test_seg_df.apply(
        librosa_resample, orig_fr=orig_fr, target_fr=target_fr
    )

    splits = [int(0.5 * len(test_seg_df))]
    cv_seg_df, eval_seg_df = np.split(
        test_seg_df.sample(frac=1, random_state=seed), splits
    )
    return train_seg_df, cv_seg_df, eval_seg_df, target_fr


class _PhoneData:
    def __init__(self, data_path, data_task, seed=0):
        # TODO: add error if not in list of all phones
        # TODO: add ability to use all vowels
        # TODO: adapt to words somehow?
        # if data_task not in all_phones:
        #     raise ValueError(f"`data_task` must be one of {all_phones}")
        data_task = sorted(data_task)
        if isinstance(data_task, str):
            dstr = data_task
        else:
            dstr = "-".join(data_task)
        task_dir = data_path / str(dstr)
        if task_dir.exists():
            self.train_phones = np.load(
                task_dir / "train_phones.npy", allow_pickle=True
            )
            self.cv_phones = np.load(task_dir / "cv_phones.npy", allow_pickle=True)
            self.eval_phones = np.load(task_dir / "eval_phones.npy", allow_pickle=True)
        else:
            train_seg_df, cv_seg_df, eval_seg_df, fs = get_train_cv_eval_dfs(
                data_path, data_task, seed=0
            )

            train_ind = train_seg_df.index.values
            cv_ind = cv_seg_df.index.values
            eval_ind = eval_seg_df.index.values

            train_vals = np.stack(train_seg_df.values)
            cv_vals = np.stack(cv_seg_df.values)
            eval_vals = np.stack(eval_seg_df.values)

            self.train_phones = train_vals
            self.cv_phones = cv_vals
            self.eval_phones = eval_vals

            task_dir.mkdir(exist_ok=True)

            np.save(task_dir / "train_phones.npy", self.train_phones)
            np.save(task_dir / "cv_phones.npy", self.cv_phones)
            np.save(task_dir / "eval_phones.npy", self.eval_phones)

            np.save(task_dir / "train_ind.npy", train_ind)
            np.save(task_dir / "cv_ind.npy", cv_ind)
            np.save(task_dir / "eval_ind.npy", eval_ind)


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
        num_data=UniformDiscrete(200, 500),  # how to choose these?
        num_target=UniformDiscrete(50, 200),  # how to choose these?
        forecast_start=UniformDiscrete(50, 300),
        device="cpu",
        subset="train",
        data_path="data/timit/TIMIT/",
        data_task=("iy"),  # add default or make default behaviour to load all phones?
    ):
        super().__init__(dtype, seed, num_tasks, batch_size, device)

        if (not isinstance(data_task, tuple)) and (data_task is not None):
            data_task = tuple(data_task)
        elif isinstance(data_task, str):
            data_task = (data_task,)
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        super().__init__(dtype, seed, num_tasks, batch_size, device=device)
        self.data_path = data_path
        self.data_task = data_task
        self.dist_x_target = convert(UniformContinuous(1, 799), AbstractDistribution)
        # ^ This doens't do anything, it's just there for sampler thing to
        # work, should fix at some point.

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
    def _load_data(data_path, data_task=("iy",)):
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
    for ds in timit_loc.iterdir():
        if (ds.is_dir()) and (ds.name in ("TRAIN", "TEST")):
            for dialect in ds.iterdir():
                if not dialect.is_dir():
                    continue
                for speaker in dialect.iterdir():
                    speaker_dirs.append(speaker)
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


def get_wav_loc(phn_row, timit_loc: Path):
    subdir = timit_loc / phn_row["dataset"] / phn_row["dialect"] / phn_row["speaker"]
    wav_loc = subdir / f"{phn_row['sentence']}.WAV"
    return wav_loc


# TODO: Do all of this ahead of time so don't need to redo for training each time.
def get_signal_data(phn_row: pd.Series, full_wav=False, normalize_gain=-20):
    wav_loc = phn_row["wav_loc"]
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


def load_phone_df(timit_loc: Path, phn0: List[str], frac=1):
    # Load list of all phones and throw and error if phn0 has values not in list
    # shuffles the dataframe as well
    speaker_dirs = get_speaker_dirs(timit_loc)
    phn0_dfs = []
    pbar = tqdm(speaker_dirs)
    for speaker_dir in pbar:
        pbar.set_description(speaker_dir.name)
        sentence_ids = get_sentence_ids(speaker_dir)
        for loc in sentence_ids:
            phn_loc = speaker_dir / f"{loc}.PHN"
            phn_df = load_phones(phn_loc)
            phn0_df0 = phn_df[np.isin(phn_df["phone"], phn0)].copy()
            phn0_df0["dialect"] = speaker_dir.parent.name.strip()
            phn0_df0["speaker"] = speaker_dir.name.strip()
            phn0_df0["sentence"] = loc
            phn0_df0["dataset"] = speaker_dir.parent.parent.name.strip()
            phn0_dfs.append(phn0_df0)
    phn0_df = pd.concat(phn0_dfs)
    phn0_df = phn0_df.reset_index(drop=True)
    phn0_df = phn0_df.sample(frac=frac)
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


def get_phone_segs(ds_df, snippet_start=0, snippet_len=800):
    """
    Get the phone segments from the dataset.
    Args:
        ds_df: dataframe with column phn_data which has the amplitudes for the frames
            from the audio file.
        snippet_start: start index of the snippet.
        snippet_len: length of the snippet.

    Returns:
        A dataframe with the phone segments.
    """
    if snippet_len == 0:
        raise ValueError("snippet_len must be greater than 0.")
    ds_frames = ds_df["phn_data"].apply(lambda x: x.shape[0])
    sm_dfs = []
    i = 0
    while True:
        start = snippet_start + i * snippet_len
        end = snippet_start + (i + 1) * snippet_len
        tf = ds_frames > end
        sm_df = ds_df.loc[tf, "phn_data"].apply(lambda x: x[start:end])
        if sm_df.shape[0] == 0:
            break
        sm_dfs.append(sm_df)
        i += 1
    phn_seg_df = pd.concat(sm_dfs)
    return phn_seg_df

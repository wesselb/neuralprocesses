import numpy as np
import torch

import pandas as pd

from lab import B
from plum import convert
from wbml.data.eeg import load_full as load_eeg

from .data import DataGenerator, apply_task
from .util import cache
from ..dist import AbstractDistribution, UniformContinuous, UniformDiscrete
from ..aggregate import AggregateInput, Aggregate

import netCDF4 as nc

from datetime import date as datetime_date

__all__ = ["AntarcticaGenerator"]


class AntarcticaGenerator(DataGenerator):
    def __init__(
        self,
        dtype,
        root_dir,
        seed=0,
        num_tasks=2**14,
        batch_size=16,
        num_sim_context=UniformDiscrete(0, 30),
        num_sim_target=UniformDiscrete(0, 30),
        min_stations=30,
        subset="train",
        device="cpu",
    ):
        super().__init__(dtype, seed, num_tasks, batch_size, device)

        self.subset = subset

        self.load_sim_data(
            root_dir=f"{root_dir}/gridded/interim/tas",
        )

        self.load_real_data(
            root_dir=f"{root_dir}/station/interim",
        )

        self.num_sim_context = num_sim_context
        self.num_sim_target = num_sim_target
        self.min_stations = min_stations

    def load_sim_data(self, root_dir):
        # Years to load
        if self.subset == "train":
            self.years = list(range(1950, 2014))

        elif self.subset == "cv":
            self.years = list(range(2014, 2018))

        elif self.subset == "eval":
            self.years = list(range(2018, 2021))

        else:
            raise ValueError("Generator mode {mode} must be 'train', 'cv' or 'eval'.")

        # Load ERA5 data
        self.sim_data = {i: nc.Dataset(f"{root_dir}/tas_{i}.nc") for i in self.years}

        # Coordinates for the ERA5 grid
        self.sim_x = self.sim_data[self.years[0]]["x"][:].data
        self.sim_y = self.sim_data[self.years[0]]["y"][:].data
        self.sim_grid_size = len(self.sim_x) * len(self.sim_x)

        # Create flattened arrays of indices for all ERA5 gridpoints
        sim_idx_x = np.arange(self.sim_x.shape[0])
        sim_idx_y = np.arange(self.sim_y.shape[0])

        self.sim_idx = np.stack(np.meshgrid(sim_idx_x, sim_idx_y), axis=0)
        self.sim_idx = np.reshape(self.sim_idx, (2, -1))

        self.sim_idx_x = self.sim_idx[0]
        self.sim_idx_y = self.sim_idx[1]

    def load_real_data(self, root_dir):
        # Load real data and metadata
        self.real_data = pd.read_csv(f"{root_dir}/all_station_data.csv")

        self.real_metadata = pd.read_csv(f"{root_dir}/all_station_metadata.csv")

        # Arrange by date to avoid slicing each time a new batch is made
        self.real_data = dict(tuple(self.real_data.groupby("date")))
        self.real_dates = np.array(
            list(
                filter(
                    lambda date: int(date[:4]) in self.years,
                    list(self.real_data.keys()),
                )
            )
        )

    def generate_batch(self):
        # Draw random number of context ERA5 datapoints
        self.state, num_sim_context = self.num_sim_context.sample(
            self.state,
            torch.int32,
        )

        # Draw random number of target ERA5 datapoints
        self.state, num_sim_target = self.num_sim_target.sample(
            self.state,
            torch.int32,
        )

        # Total number of ERA5 datapoints
        num_sim = num_sim_context + num_sim_target

        batch = {
            "sim_x": [],
            "sim_y": [],
            "sim_temp": [],
            "real_x": [],
            "real_y": [],
            "real_temp": [],
        }

        batch_size = 0
        min_real_data = None

        with B.on_device(self.device):
            while batch_size < self.batch_size:
                date = str(np.random.choice(self.real_dates))

                # Unpack station data and ERA5 data
                real_data = self.unpack_daily_real_data(self.real_data, date)

                real_x, real_y, real_temp = real_data

                if (
                    int(date[:4]) not in self.years
                    or real_x.shape[0] <= self.min_stations
                ):
                    continue

                sim_data = self.unpack_daily_sim_data(date)
                sim_x, sim_y, sim_temp = sim_data

                # Split station data into context and target
                num_real_data = real_temp.shape[0]

                if min_real_data is None or num_real_data < min_real_data:
                    min_real_data = num_real_data

                # Shuffle station data
                real_idx = np.arange(num_real_data)

                real_x = real_x[real_idx]
                real_y = real_y[real_idx]
                real_temp = real_temp[real_idx]

                # Sample ERA5 data at random
                sim_idx = np.random.choice(
                    np.arange(self.sim_grid_size),
                    replace=False,
                    size=(num_sim.cpu().numpy(),),
                )

                i = self.sim_idx_x[sim_idx]
                j = self.sim_idx_y[sim_idx]

                sim_x = self.sim_x[i]
                sim_y = self.sim_y[j]
                sim_temp = sim_temp[i, j]

                # Update batch dictionaries
                batch["sim_x"].append(sim_x)
                batch["sim_y"].append(sim_y)
                batch["sim_temp"].append(sim_temp)

                batch["real_x"].append(real_x)
                batch["real_y"].append(real_y)
                batch["real_temp"].append(real_temp)

                batch_size = batch_size + 1

            # Draw random number of context station datapoints
            num_real_context = UniformDiscrete(1, min_real_data - 1)
            self.state, num_real_context = num_real_context.sample(
                self.state, torch.int32
            )

            batch["sim_x"] = np.stack(batch["sim_x"], axis=0)
            batch["sim_y"] = np.stack(batch["sim_y"], axis=0)
            batch["sim_temp"] = np.stack(batch["sim_temp"], axis=0)

            batch["real_x"] = np.stack(
                [array[:min_real_data] for array in batch["real_x"]], axis=0
            )

            batch["real_y"] = np.stack(
                [array[:min_real_data] for array in batch["real_y"]], axis=0
            )

            batch["real_temp"] = np.stack(
                [array[:min_real_data] for array in batch["real_temp"]], axis=0
            )

            # Cast all tensors and send to device
            convert = lambda x: B.to_active_device(torch.tensor(x, dtype=self.dtype))

            scale_x = 1.2 * np.abs(self.sim_x).max()
            scale_y = 1.2 * np.abs(self.sim_y).max()
            scale_temp = 1.0

            sim_ctx_x = convert(batch["sim_x"][:, None, :num_sim_context]) / scale_x
            sim_ctx_y = convert(batch["sim_y"][:, None, :num_sim_context]) / scale_y
            sim_ctx_in = torch.tensor(
                B.concat(*[sim_ctx_x, sim_ctx_y], axis=1), dtype=self.dtype
            )
            sim_ctx_temp = (
                convert(batch["sim_temp"][:, None, :num_sim_context]) / scale_temp
            )

            sim_trg_x = convert(batch["sim_x"][:, None, num_sim_context:]) / scale_x
            sim_trg_y = convert(batch["sim_y"][:, None, num_sim_context:]) / scale_y
            sim_trg_in = torch.tensor(
                B.concat(*[sim_trg_x, sim_trg_y], axis=1), dtype=self.dtype
            )
            sim_trg_temp = (
                convert(batch["sim_temp"][:, None, num_sim_context:]) / scale_temp
            )

            if self.subset in ["cv", "eval"]:
                sim_trg_x = sim_trg_x[:, :, :0]
                sim_trg_y = sim_trg_y[:, :, :0]
                sim_trg_in = sim_trg_in[:, :, :0]
                sim_trg_temp = sim_trg_temp[:, :, :0]

                sim_trg_x = sim_trg_x[:, :, :0]
                sim_trg_y = sim_trg_y[:, :, :0]
                sim_trg_in = sim_trg_in[:, :, :0]
                sim_trg_temp = sim_trg_temp[:, :, :0]

            real_ctx_x = convert(batch["real_x"][:, None, :num_real_context]) / scale_x
            real_ctx_y = convert(batch["real_y"][:, None, :num_real_context]) / scale_y
            real_ctx_in = torch.tensor(
                B.concat(*[real_ctx_x, real_ctx_y], axis=1), dtype=self.dtype
            )
            real_ctx_temp = (
                convert(batch["real_temp"][:, None, :num_real_context]) / scale_temp
            )

            real_trg_x = convert(batch["real_x"][:, None, num_real_context:]) / scale_x
            real_trg_y = convert(batch["real_y"][:, None, num_real_context:]) / scale_y
            real_trg_in = torch.tensor(
                B.concat(*[real_trg_x, real_trg_y], axis=1), dtype=self.dtype
            )
            real_trg_temp = (
                convert(batch["real_temp"][:, None, num_real_context:]) / scale_temp
            )

            if torch.any(torch.isnan(sim_ctx_temp)):
                raise ValueError("sim_ctx_temp has nan")

            if torch.any(torch.isnan(sim_trg_temp)):
                raise ValueError("sim_trg_temp has nan")

            if torch.any(torch.isnan(real_ctx_temp)):
                raise ValueError("real_ctx_temp has nan")

            if torch.any(torch.isnan(real_trg_temp)):
                raise ValueError("real_trg_temp has nan")

            # Create context dictionary
            batch["contexts"] = [
                (sim_ctx_in, sim_ctx_temp),
                (real_ctx_in, real_ctx_temp),
            ]

            batch["xt"] = AggregateInput(
                (sim_trg_in, 0),
                (real_trg_in, 1),
            )

            batch["yt"] = Aggregate(sim_trg_temp, real_trg_temp)

            return batch

    def unpack_daily_real_data(self, all_real_data, date):
        daily_real_data = all_real_data[date]

        real_x = []
        real_y = []
        real_temp = []

        for real in daily_real_data["station"]:
            data_mask = daily_real_data["station"] == real
            metadata_mask = self.real_metadata["station"] == real

            real_data = daily_real_data[data_mask]
            real_metadata = self.real_metadata[metadata_mask]

            real_x.append(float(real_metadata["x"].values))
            real_y.append(float(real_metadata["y"].values))

            real_temp.append(float(real_data["tas"].values))

        real_x = np.array(real_x)
        real_y = np.array(real_y)
        real_temp = np.array(real_temp)

        not_nan = np.where(np.logical_not(np.isnan(real_temp)))

        real_x = real_x[not_nan]
        real_y = real_y[not_nan]
        real_temp = real_temp[not_nan]

        return real_x, real_y, real_temp

    def unpack_daily_sim_data(self, date):
        d2 = datetime_date.fromisoformat(date).toordinal()
        d1 = datetime_date.fromisoformat(f"{date[:4]}-01-01").toordinal()

        sim_temp = np.array(self.sim_data[int(date[:4])]["t2m"][d2 - d1]) - 273.15

        return self.sim_x, self.sim_y, sim_temp

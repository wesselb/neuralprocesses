import numpy as np
import torch

import pandas as pd

from lab import B
from plum import convert
from wbml.data.eeg import load_full as load_eeg

from .data import DataGenerator, apply_task
from .util import cache
from ..dist import AbstractDistribution, UniformContinuous, UniformDiscrete

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
            num_era5_context=UniformDiscrete(0, 30),
            num_era5_target=UniformDiscrete(0, 30),
            subset="train",
            mode="both",
            device="cpu",
        ):
        
        super().__init__(dtype, seed, num_tasks, batch_size, device)
        
        self.load_era5_data(
            root_dir=f"{root_dir}/gridded/interim/tas",
            mode=mode
        )
        
        self.load_station_data(
            root_dir=f"{root_dir}/station/interim",
            mode=mode
        )
        
        self.num_era5_context = num_era5_context
        self.num_era5_target = num_era5_target
            
        
    def load_era5_data(self, root_dir, mode):
        
        # Years to load
        self.years = list(range(1950, 2021))
        
        # Load ERA5 data
        self.era5_data = {
            year : nc.Dataset(f"{root_dir}/tas_{i}.nc") for i in self.years
        }
        
        # Coordinates for the ERA5 grid
        self.era5_x = self.era5_data[self.years[0]]["x"][:].data
        self.era5_y = self.era5_data[self.years[0]]["y"][:].data
        self.era5_grid_size = len(self.era5_x) * len(self.era5_x)
        
        # Create flattened arrays of indices for all ERA5 gridpoints
        era5_idx_x = np.arange(self.era5_x.shape[0])
        era5_idx_y = np.arange(self.era5_y.shape[0])
        
        self.era5_idx = np.stack(np.meshgrid(era5_idx_x, era5_idx_y), axis=0)
        self.era5_idx = np.reshape(self.era5_idx, (2, -1))
        
        self.era5_idx_x = self.era5_idx[0]
        self.era5_idx_y = self.era5_idx[1]
            
        
    def load_station_data(self, root_dir, mode):
        
        # Load station data and metadata
        self.station_data = pd.read_csv(
            f"{root_dir}/all_station_data.csv"
        )
        
        self.station_metadata = pd.read_csv(
            f"{root_dir}/all_station_metadata.csv"
        )
        
        # Arrange by date to avoid slicing each time a new batch is made
        self.station_data = dict(tuple(self.station_data.groupby("date")))
        self.station_dates = np.array(list(self.station_data.keys()))
        

    def generate_batch(self):
        
        # Draw random number of context ERA5 datapoints
        self.state, num_era5_context = self.num_era5_context.sample(
            self.state,
            np.int32
        )
        
        # Draw random number of target ERA5 datapoints
        self.state, num_era5_target = self.num_era5_target.sample(
            self.state,
            np.int32
        )
        
        # Total number of ERA5 datapoints
        num_era5 = num_era5_context + num_era5_target
        
        batch = {
            "era5_x" : [],
            "era5_y" : [],
            "era5_temperature" : [],
            "station_x" : [],
            "station_y" : [],
            "station_temperature" : [],
        }
        
        batch_size = 0
        min_station_data = None
        
        while batch_size < self.batch_size:
            
            date = np.random.choice(self.station_dates)
            
            # Unpack station data and ERA5 data
            station_data = self.unpack_daily_station_data(
                self.station_data,
                date
            )
            
            station_x, station_y, station_temperature = station_data
            
            if int(date[:4]) not in self.years or station_x.shape[0] <= 30:
                continue
            
            era5_data = self.unpack_daily_era5_data(date)
            era5_x, era5_y, era5_temperature = era5_data
            
            # Split station data into context and target
            num_station_data = station_temperature.shape[0]
            
            if min_station_data is None or num_station_data < min_station_data:
                min_station_data = num_station_data
            
            # Shuffle station data
            station_idx = np.arange(num_station_data)
            
            station_x = station_x[station_idx]
            station_y = station_y[station_idx]
            station_temperature = station_temperature[station_idx]
            
            # Sample ERA5 data at random
            era5_idx = np.random.choice(
                np.arange(self.era5_grid_size),
                replace=False,
                size=(num_era5,)
            )

            i = self.era5_idx_x[era5_idx]
            j = self.era5_idx_y[era5_idx]

            era5_x = self.era5_x[i]
            era5_y = self.era5_y[j]
            era5_temperature = era5_temperature[i, j]
            
            # Update batch dictionaries
            batch["era5_x"].append(era5_x)
            batch["era5_y"].append(era5_y)
            batch["era5_temperature"].append(era5_temperature)
            
            batch["station_x"].append(station_x)
            batch["station_y"].append(station_y)
            batch["station_temperature"].append(station_temperature)
            
            batch_size = batch_size + 1
        
        # Draw random number of context station datapoints
        num_station_context = UniformDiscrete(1, min_station_data-1)
        self.state, num_station_context = num_station_context.sample(
            self.state,
            np.int32
        )
        
        batch["era5_x"] = np.stack(batch["era5_x"], axis=0)
        batch["era5_y"] = np.stack(batch["era5_y"], axis=0)
        batch["era5_temperature"] = np.stack(batch["era5_temperature"], axis=0)
        
        batch["station_x"] = np.stack(
            [array[:min_station_data] for array in batch["station_x"]],
            axis=0
        )
        
        batch["station_y"] = np.stack(
            [array[:min_station_data] for array in batch["station_y"]],
            axis=0
        )
        
        batch["station_temperature"] = np.stack(
            [array[:min_station_data] for array in batch["station_temperature"]],
            axis=0
        )
        
        batch["era5_context_x"] = batch["era5_x"][:, :num_era5_context]
        batch["era5_context_y"] = batch["era5_y"][:, :num_era5_context]
        batch["era5_context_temperature"] = batch["era5_temperature"][:, :num_era5_context]
        
        batch["era5_target_x"] = batch["era5_x"][:, num_era5_context:]
        batch["era5_target_y"] = batch["era5_y"][:, num_era5_context:]
        batch["era5_target_temperature"] = batch["era5_temperature"][:, num_era5_context:]
        
        batch["station_context_x"] = batch["station_x"][:, :num_station_context]
        batch["station_context_y"] = batch["station_y"][:, :num_station_context]
        batch["station_context_temperature"] = batch["station_temperature"][:, :num_station_context]
        
        batch["station_target_x"] = batch["station_x"][:, num_station_context:]
        batch["station_target_y"] = batch["station_y"][:, num_station_context:]
        batch["station_target_temperature"] = batch["station_temperature"][:, num_station_context:]
        
        return batch
    

    def unpack_daily_station_data(self, all_station_data, date):
        
        daily_station_data = all_station_data[date]
        
        station_x = []
        station_y = []
        station_temperature = []
        
        for station in daily_station_data["station"]:
            
            data_mask = daily_station_data["station"] == station
            metadata_mask = self.station_metadata["station"] == station
            
            station_data = daily_station_data[data_mask]
            station_metadata = self.station_metadata[metadata_mask]
            
            station_x.append(float(station_metadata["x"].values))
            station_y.append(float(station_metadata["y"].values))
            
            station_temperature.append(float(station_data["tas"].values))
            
        station_x = np.array(station_x)
        station_y = np.array(station_y)
        station_temperature = np.array(station_temperature)
        
        return station_x, station_y, station_temperature

    
    def unpack_daily_era5_data(self, date):
        
        d2 = datetime_date.fromisoformat(date).toordinal()
        d1 = datetime_date.fromisoformat(f"{date[:4]}-01-01").toordinal()
        
        era5_temperature = np.array(
            self.era5_data[int(date[:4])]["t2m"][d2-d1]
        ) - 273.15
        
        return self.era5_x, self.era5_y, era5_temperature
        
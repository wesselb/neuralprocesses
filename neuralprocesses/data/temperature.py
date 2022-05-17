import lab as B
import netCDF4
import numpy as np
import pandas as pd

from .data import DataGenerator
from ..mask import Masked

__all__ = ["TemperatureGenerator"]


class _TemperatureData:
    def __init__(self, data_path):
        # Load the data splits.
        self.train_stations = np.load(f"{data_path}/data/train_inds.npy")[100:]
        self.cv_stations = np.load(f"{data_path}/data/train_inds.npy")[:100]
        self.eval_stations = np.load(f"{data_path}/data/test_inds.npy")

        # Load times associated with the data.
        self.times = pd.date_range("1979-01-01", "2009-01-01")[:-1]
        self.train_mask = self.times < pd.Timestamp("2000-01-01")
        self.cv_mask = pd.Timestamp("2000-01-01") <= self.times
        self.cv_mask &= self.times < pd.Timestamp("2003-01-01")
        self.eval_mask = pd.Timestamp("2003-01-01") <= self.times

        # Load the gridded data and transpose into the right form.
        # NOTE: `x_context.py` is stored with a transpose off.
        self.xc_grid = np.load(f"{data_path}/data/context/x_context.npy")
        # Here we correct for the transpose off:
        self.xc_grid = (self.xc_grid[0, :, 0:1], self.xc_grid[:, 0, 1:2])
        self.xc_grid = (self.xc_grid[0].T[None, :, :], self.xc_grid[1].T[None, :, :])
        self.yc_grid_train = np.memmap(
            f"{data_path}/data/context/y_context_training_mmap.dat",
            dtype="float32",
            mode="r",
            shape=(8766, 25, 87, 50),
        )
        self.yc_grid_eval = np.memmap(
            f"{data_path}/data/context/y_context_val_mmap.dat",
            dtype="float32",
            mode="r",
            shape=(2192, 25, 87, 50),
        )

        # Load elevation at targets and transpose into the right form.
        self.xc_elev_t = np.load(f"{data_path}/data/target/tmax_all_x_target.npy")
        self.xc_elev_t = self.xc_elev_t.T[None, :, :]
        self.yc_elev_t = np.load(f"{data_path}/data/elevation/elev_tmax_all.npy")
        self.yc_elev_t = self.yc_elev_t.T[None, :, :]

        # Load targets and transpose into the right form.
        self.xt = np.load(f"{data_path}/data/target/tmax_all_x_target.npy")
        self.xt = self.xt.T[None, :, :]
        self.yt = np.load(f"{data_path}/data/target/tmax_all_y_target.npy")
        self.yt = self.yt[:, None, :]

        # Select the relevant subset for Germany.
        lons = (6, 16)
        lats = (47, 55)

        # Process the grids.
        lon_mask = lons[0] <= self.xc_grid[0][0, 0, :]
        lon_mask &= self.xc_grid[0][0, 0, :] < lons[1]
        lat_mask = lats[0] <= self.xc_grid[1][0, 0, :]
        lat_mask &= self.xc_grid[1][0, 0, :] <= lats[1]
        self.xc_grid = (
            self.xc_grid[0][:, :, lon_mask],
            self.xc_grid[1][:, :, lat_mask],
        )
        self.yc_grid_train = self.yc_grid_train[:, :, lon_mask, :][:, :, :, lat_mask]
        self.yc_grid_eval = self.yc_grid_eval[:, :, lon_mask, :][:, :, :, lat_mask]

        # Process the elevations and the targets.
        mask = (lons[0] <= self.xt[0, 0, :]) & (self.xt[0, 0, :] < lons[1])
        mask &= (lats[0] <= self.xt[0, 1, :]) & (self.xt[0, 1, :] < lats[1])
        self.xc_elev_t = self.xc_elev_t[:, :, mask]
        self.yc_elev_t = self.yc_elev_t[:, :, mask]
        self.xt = self.xt[:, :, mask]
        self.yt = self.yt[:, :, mask]

        # Load the high-resolution elevation data.
        elev_hr = netCDF4.Dataset(f"{data_path}/elev_data_1km/data.nc")
        elev_hr_lons = elev_hr["X"][:].data
        elev_hr_lats = elev_hr["Y"][:].data

        # Select the relevant latitudes, longitudes, and elevation.
        lons_mask = (lons[0] <= elev_hr_lons) & (elev_hr_lons < lons[1])
        lats_mask = (lats[0] <= elev_hr_lats) & (elev_hr_lats < lats[1])
        elev_hr = elev_hr["topo"][lats_mask, lons_mask]
        # Extract the data, construct the mask, and save it. Note that a `False` in
        # `elev.mask` means that a data point is present!
        elev_hr_mask = B.broadcast_to(~elev_hr.mask, *B.shape(elev_hr.data))
        elev_hr_data = elev_hr.data
        elev_hr_data[elev_hr_mask == 0] = 0
        self.xc_elev_hr = (
            elev_hr_lons[lons_mask][None, None, :],
            elev_hr_lats[lats_mask][None, None, :],
        )
        # The high-resolution elevation is lat-lon form, so we need to transpose. This
        # is relatively safe, because the code will break if we get this wrong.
        # Moreover, normalise by 100 to stabilise initialisation.
        self.yc_elev_hr = B.t(elev_hr_data)[None, None, :] / 100
        self.yc_elev_hr_mask = B.t(elev_hr_mask)[None, None, :]


class TemperatureGenerator(DataGenerator):
    """Temperature generator.

    Args:
        dtype (dtype): Data type.
        seed (int, optional): Seed. Defaults to 0.
        batch_size (int, optional): Number of tasks per batch. Defaults to 16.
        target_min (int, optional): Minimum number of target points. Defaults to 5.
        target_square (float, optional): Size of the square of target points to sample.
            Defaults to not sampling a square.
        context_fraction (float, optional): Fraction of context stations. Defaults to 0.
        context_alternate (bool, optional): Alternate between sampling no contexts and
            sampling contexts. Defaults to `False`.
        subset (str, optional): Subset of the data. Must be one of `"train"`, `"cv"` or
            `"eval"`. Defaults to `"train"`.
        passes (int, optional): How many times to cycle through the data in an epoch.
            Defaults to 1.
        device (str, optional): Device. Defaults to `"cpu"`.
        data_path (str, optional): Path to the data. Defaults to `"climate_data"`.

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of the data type with 64 bytes.
        int64 (dtype): Integral version of the data type with 64 bytes.
        seed (int): Seed.
        batch_size (int): Number of tasks per batch.
        num_batches (int): Number of batches in an epoch.
        target_min (int): Minimum number of target points.
        target_square (float): Size of the square of target points to sample.
        context_fraction (float): Fraction of context stations.
        context_alternate (bool): Alternate between sampling no contexts and sampling
            contexts.
        passes (int): How many times to cycle through the data in an epoch.
        device (str): Device.
    """

    _data = None

    def __init__(
        self,
        dtype,
        seed=0,
        batch_size=16,
        target_min=5,
        target_square=0.0,
        context_fraction=0.0,
        context_alternate=False,
        subset="train",
        passes=1,
        device="cpu",
        data_path="climate_data",
    ):
        self.target_min = target_min
        self.target_square = target_square
        self.context_fraction = context_fraction
        self.context_alternate = context_alternate
        self._alternate_i = 0
        self.passes = passes

        # Load data if it isn't yet loaded.
        if TemperatureGenerator._data is None:
            TemperatureGenerator._data = _TemperatureData(data_path)
        data = TemperatureGenerator._data

        if subset == "train":
            num_tasks = data.train_mask.sum()
            self._times = data.times[data.train_mask]
            n = data.yc_grid_train.shape[0]
            self._xc_grid = data.xc_grid
            self._yc_grid = data.yc_grid_train[data.train_mask[:n]]
            self._xc_elev_t = data.xc_elev_t[:, :, data.train_stations]
            self._yc_elev_t = data.yc_elev_t[:, :, data.train_stations]
            self._xc_elev_hr = data.xc_elev_hr
            self._yc_elev_hr = data.yc_elev_hr
            self._yc_elev_hr_mask = data.yc_elev_hr_mask
            self._xt = data.xt[:, :, data.train_stations]
            self._yt = data.yt[:, :, data.train_stations][data.train_mask]

        elif subset == "cv":
            num_tasks = data.cv_mask.sum()
            self._times = data.times[data.cv_mask]
            n = data.yc_grid_train.shape[0]
            self._xc_grid = data.xc_grid
            self._yc_grid = data.yc_grid_train[data.cv_mask[:n]]
            self._xc_elev_t = data.xc_elev_t[:, :, data.cv_stations]
            self._yc_elev_t = data.yc_elev_t[:, :, data.cv_stations]
            self._xc_elev_hr = data.xc_elev_hr
            self._yc_elev_hr = data.yc_elev_hr
            self._yc_elev_hr_mask = data.yc_elev_hr_mask
            self._xt = data.xt[:, :, data.cv_stations]
            self._yt = data.yt[:, :, data.cv_stations][data.cv_mask]

        elif subset == "eval":
            num_tasks = data.eval_mask.sum()
            self._times = data.times[data.eval_mask]
            self._xc_grid = data.xc_grid
            self._yc_grid = data.yc_grid_eval
            self._xc_elev_t = data.xc_elev_t[:, :, data.eval_stations]
            self._yc_elev_t = data.yc_elev_t[:, :, data.eval_stations]
            self._xc_elev_hr = data.xc_elev_hr
            self._yc_elev_hr = data.yc_elev_hr
            self._yc_elev_hr_mask = data.yc_elev_hr_mask
            self._xt = data.xt[:, :, data.eval_stations]
            self._yt = data.yt[:, :, data.eval_stations][data.eval_mask]

        else:
            raise ValueError(f'Invalid subset "{subset}".')

        super().__init__(dtype, seed, num_tasks, batch_size, device)

        # Setup the first shuffle.
        self.shuffle()

    def shuffle(self):
        """Shuffle the data, preparing for a new epoch."""
        perms = []
        for _ in range(self.passes):
            self.state, perm = B.randperm(self.state, self.int64, len(self._times))
            perms.append(perm)
        self._inds = B.concat(*perms, axis=0)

    def generate_batch(self):
        if len(self._inds) == 0:
            raise RuntimeError("No data left. Shuffle the generator and try again.")

        # Collect tasks.
        tasks = []
        while len(tasks) < self.batch_size:
            if len(self._inds) == 0:
                break

            # Take the current index.
            i = self._inds[0]
            self._inds = self._inds[1:]

            tasks.append(
                {
                    "xc_grid_lons": self._xc_grid[0][0],
                    "xc_grid_lats": self._xc_grid[1][0],
                    "yc_grid": self._yc_grid[i],
                    "xc_elev_t": self._xc_elev_t[0],
                    "yc_elev_t": self._yc_elev_t[0],
                    "xc_elev_hr_lons": self._xc_elev_hr[0][0],
                    "xc_elev_hr_lats": self._xc_elev_hr[1][0],
                    "yc_elev_hr": self._yc_elev_hr[0],
                    "yc_elev_hr_mask": self._yc_elev_hr_mask[0],
                    "xt": self._xt[0],
                    "yt": self._yt[i],
                }
            )

        # Stack tasks into one batch and convert to the right framework.
        b = {
            k: B.cast(self.dtype, B.stack(*(t[k] for t in tasks)))
            for k in tasks[0].keys()
        }

        # Check if it is our turn.
        alternate_turn = self._alternate_i % 2 == 0
        self._alternate_i += 1

        # Perform a division into context and target.
        n = B.shape(b["xt"], -1)
        if not self.context_alternate or (self.context_alternate and alternate_turn):
            nc_upper = max(int(self.context_fraction * n), 1)
        else:
            nc_upper = 1
        self.state, nc = B.randint(self.state, self.int64, lower=0, upper=nc_upper)
        self.state, perm = B.randperm(self.state, self.int64, n)
        b["xc_s"] = B.take(b["xt"], perm[:nc], axis=-1)
        b["yc_s"] = B.take(b["yt"], perm[:nc], axis=-1)
        b["xt"] = B.take(b["xt"], perm[nc:], axis=-1)
        b["yt"] = B.take(b["yt"], perm[nc:], axis=-1)

        # Apply the mask to the station contexts, which have only one channel.
        mask = ~B.isnan(b["yc_s"])
        b["yc_s"] = Masked(B.where(mask, b["yc_s"], B.zero(b["yc_s"])), mask)

        # Determine bounds of the target points for the square selection.
        lowers = B.min(B.min(b["xt"], axis=2), axis=0)
        uppers = B.max(B.max(b["xt"], axis=2), axis=0)

        if self.target_square > 0:
            # Sample a square.
            half_side = self.target_square / 2
            lowers_wide = lowers - self.target_square
            uppers_wide = uppers + self.target_square

            while True:
                # Sample a random centre of the square in a way that every target point
                # has the same probability of being selected. For this, we use to
                # widened lower and upper bounds
                self.state, rand = B.rand(self.state, self.dtype, 2)
                centre = lowers_wide + rand * (uppers_wide - lowers_wide)

                # Select the targets within the square.
                mask = B.all(
                    B.all(
                        (b["xt"] >= centre[None, :, None] - half_side)
                        & (b["xt"] <= centre[None, :, None] + half_side),
                        axis=1,
                    ),
                    axis=0,
                )

                # Only stop sampling if the minimum number of targets was selected.
                if B.sum(mask) >= self.target_min:
                    b["xt"] = B.take(b["xt"], mask, axis=-1)
                    b["yt"] = B.take(b["yt"], mask, axis=-1)
                    break

        # Move everything to the right device.
        with B.on_device(self.device):
            b = {k: B.to_active_device(v) for k, v in b.items()}

        # Finally, construct the composite context.
        b["contexts"] = [
            (b["xc_s"], b["yc_s"]),
            ((b["xc_grid_lons"], b["xc_grid_lats"]), b["yc_grid"]),
            (
                (b["xc_elev_hr_lons"], b["xc_elev_hr_lats"]),
                Masked(b["yc_elev_hr"], b["yc_elev_hr_mask"]),
            ),
        ]

        return b

    def epoch(self):
        self.shuffle()
        return DataGenerator.epoch(self)

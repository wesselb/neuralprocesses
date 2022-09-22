import lab as B
import netCDF4
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .data import DataGenerator
from .util import cache
from ..augment import AugmentedInput
from ..dist import TruncatedGeometric
from ..mask import Masked

__all__ = ["TemperatureGenerator"]


class _TemperatureData:
    def __init__(
        self,
        data_path,
        data_task,
        data_fold,
        cv_stations,
        context_elev_hr,
        target_elev_interpolate,
    ):
        if data_task not in {"germany", "europe", "value"}:
            raise ValueError(
                f'`data_task` must be one of "germany", "europe", or "value".'
            )

        # Load the data splits.
        if data_task == "germany":
            # For Germany, the split is predetermined.
            self.train_stations = np.load(f"{data_path}/data/train_inds.npy")
            self.cv_stations = np.load(f"{data_path}/data/train_inds.npy")
            if cv_stations:
                # We agree to use the first 100 for cross-validation.
                self.cv_stations = self.cv_stations[:100]
                self.train_stations = self.train_stations[100:]
            self.eval_stations = np.load(f"{data_path}/data/test_inds.npy")
        elif data_task == "value":
            # For VALUE, we evaluate on the same stations that we train on, so we don't
            # cross-validate on other stations.
            self.train_stations = slice(None, None, None)
            self.cv_stations = slice(None, None, None)
            self.eval_stations = slice(None, None, None)
        elif data_task == "europe":
            # For the variant of VALUE, we train on different stations.
            if cv_stations:
                # This split is not predetermined, so we choose a random one here.
                n = 3043
                n_train = int(n * 0.85)
                # The seed below should not be altered! NumPy's `RandomState` policy
                # says that this should always produce the exact same permutation for
                # the same seed.
                state = B.create_random_state(np.int64, seed=99)
                _, perm = B.randperm(state, np.int64, n)
                self.train_stations = perm[:n_train]
                self.cv_stations = perm[n_train:]
            else:
                self.train_stations = slice(None, None, None)
                self.cv_stations = slice(None, None, None)
            self.eval_stations = slice(None, None, None)
        else:  # pragma: no cover
            # This can never be reached.
            raise RuntimeError(f'Bad data task "{data_task}".')

        # Load times associated with the data.
        if data_fold not in {1, 2, 3, 4, 5}:
            raise ValueError("`data_fold` must be a number between 1 and 5.")
        self.times = pd.date_range("1979-01-01", "2009-01-01")[:-1]
        _pdt = pd.Timestamp
        folds = [
            (_pdt("1979-01-01") <= self.times) & (self.times < _pdt("1985-01-01")),
            (_pdt("1985-01-01") <= self.times) & (self.times < _pdt("1991-01-01")),
            (_pdt("1991-01-01") <= self.times) & (self.times < _pdt("1997-01-01")),
            (_pdt("1997-01-01") <= self.times) & (self.times < _pdt("2003-01-01")),
            (_pdt("2003-01-01") <= self.times) & (self.times < _pdt("2009-01-01")),
        ]
        # `data_fold` starts at 1 rather than 0.
        train_folds = [fold for i, fold in enumerate(folds) if i != data_fold - 1]
        self.train_mask = np.logical_or.reduce(train_folds)
        self.eval_mask = folds[data_fold - 1]
        # Take the last 1000 days (a little under three years) for cross-validation.
        inds = set(np.where(self.train_mask)[0][-1000:])
        self.cv_mask = np.array([i in inds for i in range(len(self.train_mask))])
        # Cancel the cross-validation in the training mask.
        self.train_mask = self.train_mask & ~self.cv_mask

        # Load the gridded data and transpose into the right form.
        if data_task == "germany":
            # NOTE: `x_context.py` is stored with a transpose off.
            self.xc_grid = np.load(f"{data_path}/data/context/x_context.npy")
            # Here we correct for the transpose off. Note the colon comes second in the
            # longitudes and first in the latitudes.
            self.xc_grid = (
                self.xc_grid[0, :, 0:1].T[None, :, :],
                self.xc_grid[:, 0, 1:2].T[None, :, :],
            )
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
            self.yc_grid = B.concat(self.yc_grid_train, self.yc_grid_eval, axis=0)
        elif data_task in {"europe", "value"}:
            self.xc_grid = np.load(
                f"{data_path}/data/context/x_context_coarse_final.npy"
            )
            self.xc_grid = (
                self.xc_grid[:, 0, 0:1].T[None, :, :],
                self.xc_grid[0, :, 1:2].T[None, :, :],
            )
            self.yc_grid = np.load(
                f"{data_path}/data/context/y_context_coarse_final.npy",
                mmap_mode="r",
            )
        else:  # pragma: no cover
            # This can never be reached.
            raise RuntimeError(f'Bad data task "{data_task}".')

        # Load targets and transpose into the right form.
        if data_task in {"germany", "europe"}:
            self.xt = np.load(f"{data_path}/data/target/tmax_all_x_target.npy")
            self.xt = self.xt.T[None, :, :]
            self.yt = np.load(f"{data_path}/data/target/tmax_all_y_target.npy")
            self.yt = self.yt[:, None, :]

            # Load elevation at targets and transpose into the right form.
            self.xt_elev = np.load(f"{data_path}/data/target/tmax_all_x_target.npy")
            self.xt_elev = self.xt_elev.T[None, :, :]
            self.yt_elev = np.load(f"{data_path}/data/elevation/elev_tmax_all.npy")
            # We just use the elevation and ignore the other two features.
            self.yt_elev = self.yt_elev.T[None, :1, :]
        elif data_task == "value":
            self.xt = np.load(f"{data_path}/data/target/value_x_target.npy")
            self.xt = self.xt.T[None, :, :]
            self.yt = np.load(f"{data_path}/data/target/tmax_value_y_target.npy")
            self.yt = self.yt[:, None, :]
            mask = (
                # The target values go up to 2011, but we only need up to 2009.
                pd.date_range("1979-01-01", "2011-01-01")[:-1]
                < pd.Timestamp("2009-01-01")
            )
            self.yt = self.yt[mask]

            # Load elevation at targets and transpose into the right form.
            self.xt_elev = np.load(f"{data_path}/data/target/value_x_target.npy")
            self.xt_elev = self.xt_elev.T[None, :, :]
            self.yt_elev = np.load(f"{data_path}/data/elevation/elev_value.npy")
            # We just use the elevation and ignore the other two features.
            self.yt_elev = self.yt_elev.T[None, :1, :]
        else:  # pragma: no cover
            # This can never be reached.
            raise RuntimeError(f'Bad data task "{data_task}".')

        # Select the relevant subset of the data.
        if data_task == "germany":
            # For Germany, these bounds are chosen to match the predetermined train-test
            # split. The bounds can therefore not be altered!
            lons = (6, 16)
            lats = (47, 55)
            assert_no_data_lost = False
        elif data_task in {"europe", "value"}:
            # These bounds must cover all target stations; otherwise, the train-test
            # split will not line up.
            lons = (-24, 40)
            lats = (35, 75)
            assert_no_data_lost = True
        else:  # pragma: no cover
            # This can never be reached.
            raise RuntimeError(f'Bad data task "{data_task}".')

        # Process the grids.
        lon_mask = lons[0] <= self.xc_grid[0][0, 0, :]
        lon_mask &= self.xc_grid[0][0, 0, :] < lons[1]
        lat_mask = lats[0] <= self.xc_grid[1][0, 0, :]
        lat_mask &= self.xc_grid[1][0, 0, :] <= lats[1]
        if assert_no_data_lost and (B.any(~lon_mask) or B.any(~lat_mask)):
            raise AssertionError("Longitude and latitude bounds are too tight.")
        self.xc_grid = (
            self.xc_grid[0][:, :, lon_mask],
            self.xc_grid[1][:, :, lat_mask],
        )
        self.yc_grid = self.yc_grid[:, :, lon_mask, :][:, :, :, lat_mask]

        # Process the elevations and the targets.
        mask = (lons[0] <= self.xt[0, 0, :]) & (self.xt[0, 0, :] < lons[1])
        mask &= (lats[0] <= self.xt[0, 1, :]) & (self.xt[0, 1, :] < lats[1])
        if assert_no_data_lost and B.any(~mask):
            raise AssertionError("Longitude and latitude bounds are too tight.")
        self.xt = self.xt[:, :, mask]
        self.yt = self.yt[:, :, mask]
        self.xt_elev = self.xt_elev[:, :, mask]
        self.yt_elev = self.yt_elev[:, :, mask]

        if context_elev_hr:
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
            # The high-resolution elevation is lat-lon form, so we need to transpose.
            # This is relatively safe, because the code will break if we get this wrong.
            self.yc_elev_hr = B.transpose(elev_hr_data)[None, None, :]
            self.yc_elev_hr_mask = B.transpose(elev_hr_mask)[None, None, :]

        if target_elev_interpolate:
            # First, we reshape the grid data in a form that `griddata` expects.
            z = self.yc_elev_hr[0, 0]
            x = B.flatten(self.xc_elev_hr[0])
            y = B.flatten(self.xc_elev_hr[1])
            x = B.broadcast_to(x[:, None], *B.shape(z))
            y = B.broadcast_to(y[None, :], *B.shape(z))
            xy = B.stack(B.flatten(x), B.flatten(y), axis=1)
            # Perform bilinear interpolation to `self.xt_elev`.
            self.yt_elev = griddata(
                xy,
                B.flatten(z),
                B.transpose(self.xt_elev[0]),
            )[None, None, :]


class TemperatureGenerator(DataGenerator):
    """Temperature generator.

    Args:
        dtype (dtype): Data type.
        seed (int, optional): Seed. Defaults to 0.
        batch_size (int, optional): Number of tasks per batch. Defaults to 16.
        cv_stations (bool, optional): Cross-validate on different stations. Defaults
            to `True`.
        context_sample (bool, optional): Randomly split the data into context and
            target. Defaults to `False`.
        context_sample_factor (scalar, optional): When randomly splitting the data into
            context and target, emphasise the lower numbers more. This factor is the
            probability of the lowest number divided by the probability of
            `context_sample_factor_at`, if it is given, and otherwise the highest
            number.
        context_sample_factor_at (scalar, optional): Upper bound for
            `context_sample_factor`.
        context_elev_hr (bool, optional): Load the high-resolution elevation data as
            a context set. If set to `False`, that context set will be `(None, None)`.
            Defaults to `True`.
        target_min (int, optional): Minimum number of target points. Defaults to 5.
        target_max (int, optional): Maximum number of target points. Defaults to no
            maximum.
        target_dropnan (int, optional): Drop stations which have NaNs. Defaults to
            `False`.
        target_square (float, optional): Size of the square of target points to sample.
            Defaults to not sampling a square.
        target_elev (bool, optional): Append the elevation at the target inputs as
            auxiliary information. Defaults to `False`.
        target_elev_interpolate (bool, optional): Estimate the elevation at the target
            inputs by bilinearly interpolating the elevation on the high-resolution
            1 km grid. Defaults to `False`.
        subset (str, optional): Subset of the data. Must be one of `"train"`, `"cv"` or
            `"eval"`. Defaults to `"train"`.
        passes (int, optional): How many times to cycle through the data in an epoch.
            Defaults to 1.
        data_task (str, optional): Task. Must be one of `"germany"`, `"europe"`, or
            `"value"`. Defaults to `"germany"`.
        data_fold (int, optional): Fold. Must be a number between 1 and 5. Defauls to 5.
        data_path (str, optional): Path to the data. Defaults to `"climate_data"`.
        device (str, optional): Device. Defaults to `"cpu"`.

    Attributes:
        dtype (dtype): Data type.
        float64 (dtype): Floating point version of the data type with 64 bytes.
        int64 (dtype): Integral version of the data type with 64 bytes.
        seed (int): Seed.
        batch_size (int): Number of tasks per batch.
        num_batches (int): Number of batches in an epoch.
        context_sample (bool): Randomly split the data into context and target.
        context_sample_factor (scalar): When randomly splitting the data into context
            and target, emphasise the lower numbers more. This factor is the probability
            of the lowest number divided by the probability of
            `context_sample_factor_at`, if it is given, and otherwise the highest
            number.
        context_sample_factor_at (scalar): Upper bound for `context_sample_factor`.
        target_min (int): Minimum number of target points.
        target_max (int, optional): Maximum number of target points.
        target_dropnan (int, optional): Drop stations which have NaNs.
        target_square (float): Size of the square of target points to sample.
        target_elev (bool): Append the elevation at the target inputs as auxiliary
            information.
        passes (int): How many times to cycle through the data in an epoch.
        data (:class:`neuralprocesses.data.temperature._TemperatureData`):
            The raw data.
        device (str): Device.
    """

    _data_cache = {}

    def __init__(
        self,
        dtype,
        seed=0,
        batch_size=16,
        cv_stations=True,
        context_sample=False,
        context_sample_factor=10,
        context_sample_factor_at=None,
        context_elev_hr=True,
        target_min=5,
        target_max=None,
        target_dropnan=False,
        target_square=0.0,
        target_elev=False,
        target_elev_interpolate=False,
        subset="train",
        passes=1,
        device="cpu",
        data_task="germany",
        data_fold=5,
        data_path="climate_data",
    ):
        self.context_sample = context_sample
        self.context_sample_factor = context_sample_factor
        self.context_sample_factor_at = context_sample_factor_at
        self.context_elev_hr = context_elev_hr
        self.target_min = target_min
        self.target_max = target_max
        self.target_dropnan = target_dropnan
        self.target_square = target_square
        self.target_elev = target_elev
        self._alternate_i = 0
        self.passes = passes

        data = TemperatureGenerator._load_data(
            data_path=data_path,
            data_task=data_task,
            data_fold=data_fold,
            cv_stations=cv_stations,
            context_elev_hr=context_elev_hr,
            target_elev_interpolate=target_elev_interpolate,
        )
        # Expose the raw data to the user.
        self.data = data

        if subset == "train":
            mask = data.train_mask
            stations = data.train_stations
        elif subset == "cv":
            mask = data.cv_mask
            stations = data.cv_stations
        elif subset == "eval":
            mask = data.eval_mask
            stations = data.eval_stations
        else:
            raise ValueError(f'Invalid subset "{subset}".')

        num_tasks = mask.sum()
        self._mask = mask
        self._times = data.times[mask]
        self._xc_grid = data.xc_grid
        self._yc_grid = data.yc_grid[mask]
        if context_elev_hr:
            self._xc_elev_hr = data.xc_elev_hr
            self._yc_elev_hr = data.yc_elev_hr
            self._yc_elev_hr_mask = data.yc_elev_hr_mask
        self._xc_elev_station = data.xt_elev
        self._yc_elev_station = data.yt_elev
        self._xt = data.xt[:, :, stations]
        self._yt = data.yt[:, :, stations][mask]
        self._xt_elev = data.xt_elev[:, :, stations]
        self._yt_elev = data.yt_elev[:, :, stations]

        super().__init__(dtype, seed, num_tasks, batch_size, device)

        # Setup the first shuffle.
        self.shuffle()

    @staticmethod
    @cache
    def _load_data(
        data_path,
        data_task,
        data_fold,
        cv_stations,
        context_elev_hr,
        target_elev_interpolate,
    ):
        return _TemperatureData(
            data_path=data_path,
            data_task=data_task,
            data_fold=data_fold,
            cv_stations=cv_stations,
            context_elev_hr=context_elev_hr,
            target_elev_interpolate=target_elev_interpolate,
        )

    def shuffle(self):
        """Shuffle the data, preparing for a new epoch."""
        perms = []
        for _ in range(self.passes):
            self.state, perm = B.randperm(self.state, self.int64, len(self._times))
            perms.append(perm)
        self._inds = B.concat(*perms, axis=0)

    def generate_batch(self, nc=None):
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

            task = {
                "xc_grid_lons": self._xc_grid[0],
                "xc_grid_lats": self._xc_grid[1],
                "yc_grid": self._yc_grid[i : i + 1],
                "xc_elev_station": self._xc_elev_station,
                "yc_elev_station": self._yc_elev_station,
                "xt": self._xt,
                "yt": self._yt[i : i + 1],
                "yt_elev": self._yt_elev,
            }
            if self.context_elev_hr:
                task = dict(
                    task,
                    **{
                        "xc_elev_hr_lons": self._xc_elev_hr[0],
                        "xc_elev_hr_lats": self._xc_elev_hr[1],
                        "yc_elev_hr": self._yc_elev_hr,
                        "yc_elev_hr_mask": self._yc_elev_hr_mask,
                    },
                )
            tasks.append(task)

        def _concat(*xs):
            if all(id(xs[0]) == id(x) for x in xs):
                # No need to cast, convert, and concatenate all of them. This is much
                # more efficient.
                x = B.cast(self.dtype, xs[0])
                return B.tile(x, len(xs), *((1,) * (B.rank(x) - 1)))
            else:
                return B.concat(*(B.cast(self.dtype, x) for x in xs), axis=0)

        # Concatenate tasks into one batch and convert to the right framework.
        b = {k: _concat(*(t[k] for t in tasks)) for k in tasks[0].keys()}

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
                available = B.cast(
                    B.dtype_float(mask),
                    ~B.isnan(B.take(b["yt"], mask, axis=-1)),
                )
                if B.min(B.sum(available, axis=(1, 2))) >= self.target_min:
                    b["xc_s_outside_square"] = B.take(b["xt"], ~mask, axis=-1)
                    b["yc_s_outside_square"] = B.take(b["yt"], ~mask, axis=-1)
                    b["xt"] = B.take(b["xt"], mask, axis=-1)
                    b["yt"] = B.take(b["yt"], mask, axis=-1)
                    b["yt_elev"] = B.take(b["yt_elev"], mask, axis=-1)
                    break

        else:
            # We don't sample a square, so nothing is outside the square.
            b["xc_s_outside_square"] = b["xt"][:, :, :0]
            b["yc_s_outside_square"] = b["yt"][:, :, :0]

        if self.context_sample:
            # Perform a division into context and target. In the line below, `True`
            # indicates that the index belongs to a point inside the square and `False`
            # indicates that the index belongs to a point outside the square.
            inds = [(True, i) for i in range(B.shape(b["xt"], -1))]
            inds += [(False, i) for i in range(B.shape(b["xc_s_outside_square"], -1))]
            # Shuffle the points.
            self.state, perm = B.randperm(self.state, self.int64, len(inds))
            inds = [inds[i] for i in perm]
            if nc is None:
                # Find the maximum number of context points by ensuring that there are
                # at least `self.target_min` in the target set.
                # TODO: Is the below loop the right way of doing this?
                nc_upper = len(inds)
                count = 0
                for inside, i in reversed(inds):
                    # If the current point is inside the square and if at least one
                    # observation is available for all stations, only then increase
                    # the count.
                    if inside and B.all(B.any(~B.isnan(b["yt"][:, :, i]), axis=1)):
                        count += 1
                    nc_upper -= 1
                    if count >= self.target_min:
                        break
                # Now sample from a truncated geometric distribution, which has the
                # ability to emphasise the lower context numbers.
                dist = TruncatedGeometric(
                    0,
                    nc_upper,
                    self.context_sample_factor,
                    self.context_sample_factor_at,
                )
                self.state, nc = dist.sample(self.state, self.int64)
            inds_c_inside = [i for inside, i in inds[:nc] if inside]
            inds_t_inside = [i for inside, i in inds[nc:] if inside]
            inds_c_outside = [i for inside, i in inds[:nc] if not inside]

            # Perform the split.
            b["xc_s"] = B.concat(
                B.take(b["xt"], inds_c_inside, axis=-1),
                B.take(b["xc_s_outside_square"], inds_c_outside, axis=-1),
                axis=-1,
            )
            b["yc_s"] = B.concat(
                B.take(b["yt"], inds_c_inside, axis=-1),
                B.take(b["yc_s_outside_square"], inds_c_outside, axis=-1),
                axis=-1,
            )
            b["xt"] = B.take(b["xt"], inds_t_inside, axis=-1)
            b["yt"] = B.take(b["yt"], inds_t_inside, axis=-1)
            b["yt_elev"] = B.take(b["yt_elev"], inds_t_inside, axis=-1)

            # Apply the mask to the station contexts, which have only one channel.
            mask = ~B.isnan(b["yc_s"])
            b["yc_s"] = Masked(B.where(mask, b["yc_s"], B.zero(b["yc_s"])), mask)

        else:
            # There is no context to sample.
            b["xc_s"] = b["xt"][:, :, :0]
            b["yc_s"] = b["yt"][:, :, :0]

        # Move everything to the right device.
        with B.on_device(self.device):
            b = {k: B.to_active_device(v) for k, v in b.items()}

        # Finally, construct the composite context.
        b["contexts"] = [
            (b["xc_s"], b["yc_s"]),
            ((b["xc_grid_lons"], b["xc_grid_lats"]), b["yc_grid"]),
            # For the elevation, use a helpful normalisation.
            (
                (
                    (b["xc_elev_hr_lons"], b["xc_elev_hr_lats"]),
                    Masked(b["yc_elev_hr"] / 100, b["yc_elev_hr_mask"]),
                )
                if self.context_elev_hr
                else (None, None)
            ),
            (b["xc_elev_station"], b["yc_elev_station"] / 100),
        ]

        # Drop stations which have NaNs, if asked for.
        if self.target_dropnan:
            mask = ~B.any(B.any(B.isnan(b["yt"]), axis=0), axis=0)
            b["xt"] = B.take(b["xt"], mask, axis=-1)
            b["yt"] = B.take(b["yt"], mask, axis=-1)
            b["yt_elev"] = B.take(b["yt_elev"], mask, axis=-1)

        # Ensure that the maximum number of targets isn't exceeded.
        if self.target_max:
            self.state, perm = B.randperm(self.state, self.int64, B.shape(b["xt"], -1))
            inds = perm[: self.target_max]
            b["xt"] = B.take(b["xt"], inds, axis=-1)
            b["yt"] = B.take(b["yt"], inds, axis=-1)
            b["yt_elev"] = B.take(b["yt_elev"], inds, axis=-1)

        # Append the elevation as auxiliary information, if asked for.
        if self.target_elev:
            b["xt"] = AugmentedInput(b["xt"], b["yt_elev"])

        return b

    def epoch(self):
        self.shuffle()
        return DataGenerator.epoch(self)

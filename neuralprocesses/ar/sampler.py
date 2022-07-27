import argparse
import concurrent.futures
import logging
import os
from collections import namedtuple
from enum import Enum
from functools import cached_property
from pathlib import Path
from scipy.special import logsumexp
import subprocess

import h5py
import lab as B
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from scipy.stats import norm
from tqdm import tqdm

import experiment as exp
from neuralprocesses import torch as nps
from neuralprocesses.dist import UniformDiscrete, UniformContinuous, Grid
from neuralprocesses.ar.trajectory import construct_trajectory_gens
from neuralprocesses.ar.trajectory import AbstractTrajectoryGenerator
from neuralprocesses.aggregate import AggregateInput

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in np.arange(0, len(l), group_size):
        yield l[i:i+group_size]


class Groups(Enum):
    MARGINAL_DENSITIES = "marginal_densities"
    TRAJECTORIES = "trajectories"


class Datasets(Enum):
    LOG_LIKELIHOODS = "log_likelihoods"
    MEANS = "means"
    VARIANCES = "variances"


def read_hdf5(hdf5_loc: Path, group_name: str):
    if not hdf5_loc.exists():
        raise FileNotFoundError(f"{hdf5_loc} does not exist")
    ss = FunctionTrajectorySet(hdf5_loc=hdf5_loc, group_name=group_name)
    return ss


def get_function_context(contexts: torch.Tensor, i: int) -> torch.Tensor:
    """
    Get context for the i-th function the batch of contexts.

    Args:
        contexts: the contexts for all functions in the batch.
        i: index of the function
    Returns:
        the context for the i-th function.
    """
    xc = contexts[0][0][i, :, :].reshape(1, 1, -1)
    yc = contexts[0][1][i, :, :].reshape(1, 1, -1)
    contexts_i = [(xc, yc)]
    return contexts_i


def get_func_expected_ll(lh0) -> float:
    """
    Get the log likelihood of for one function with trajectories and multiple target points
    Args:
        lh0: array with shape (num_target_points, trajectory_length)

    Returns:
        ll:
    """
    # Get gmms from component gaussian likelihooods
    # mn = lh0.mean(axis=1)
    # # Get the log of this value for log likelihood
    # Sum the log likelihoods for each target point
    # (not using chain rule to factorize, just summing)
    # We are only assessing quality of marginals here, not the joint.

    num_components = lh0.shape[1]
    num_targets = lh0.shape[0]
    v2 = logsumexp(lh0, axis=1).sum()
    v1 = num_targets * np.log(num_components)
    expected_ll1 = v2 - v1

    # mn = np.nanmean(B.exp(lh0), axis=1)
    # target_lls = np.log(mn)
    # expected_ll2 = target_lls.sum()
    # print(np.isclose(expected_ll1, expected_ll2))

    return expected_ll1


class TrajectorySet:
    def __init__(
        self,
        density_loc: Path,
        model,
        data_generator,
        trajectory_generator: AbstractTrajectoryGenerator,
        num_functions_per_context_size=10,
        num_trajectories=100,
        context_sizes=(1,), # just use one by default
        device="cpu",
        overwrite=True,
        metadata=None,
        load=False,
    ):
        # self. = density_loc
        self.model = model
        self.tqdm = tqdm
        # these gens need to be retrieved somehow when loading from hdf5
        # TODO: batch won't be defined at this level, will depend on the SampleSet
        self.overwrite = overwrite
        self.num_functions_per_context_size = num_functions_per_context_size
        self.num_trajectories = num_trajectories
        self.metadata = metadata
        self.load = load

        self._function_trajectory_sets = []
        self._xt = None
        self._yt = None
        self._density_loc = None
        self._num_functions = None
        self._context_sizes = None
        self._num_targets = None
        self._data_generator = None
        self._trajectory_generator = None
        self._trajectory_length = None

        # properties
        self.context_sizes = context_sizes
        self.device = device
        self.data_generator = data_generator
        self.trajectory_generator = trajectory_generator
        self.density_loc = density_loc

        # TODO: num context won't be defined at this level, will depend on the SampleSet
        self.num_density_eval_locations = None

    @classmethod
    def from_hdf5(cls, density_loc):
        with h5py.File(density_loc, "r") as f:
            num_functions_per_context_size = f.attrs["num_functions_per_context_size"]
            num_trajectories = f.attrs["num_trajectories"]
            context_sizes = f.attrs["context_sizes"]
        context_sizes = np.arange(num_functions_per_context_size)
        context_sizes
        return cls(
            density_loc,
            model=None,
            data_generator=None,
            trajectory_generator=None,
            num_functions_per_context_size=num_functions_per_context_size,
            num_trajectories=num_trajectories,
            context_sizes=context_sizes,
            load=True
        )

    @property
    def data_generator(self):
        return self._data_generator

    @data_generator.setter
    def data_generator(self, data_generator):
        self._data_generator = data_generator
        if data_generator is None:
            LOG.warning("Model is frozen if data_generator is None.")
        else:
            # assumes upper and lower are the same, only one number for num_targets
            self._num_targets = self.data_generator.num_target.upper
            self._xt = torch.Tensor(self.num_functions, 1, self.num_targets)
            self._yt = torch.Tensor(self.num_functions, 1, self.num_targets)

    @property
    def trajectory_generator(self):
        return self._trajectory_generator

    @trajectory_generator.setter
    def trajectory_generator(self, trajectory_generator):
        self._trajectory_generator = trajectory_generator
        if trajectory_generator is None:
            LOG.warning("Model is frozen if trajectory_generator is None.")
        else:
            self._trajectory_length = self.trajectory_generator.trajectory_length

    @property
    def trajectory_length(self):
        return self._trajectory_length

    @property
    def num_targets(self):
        return self._num_targets

    @property
    def xt(self):
        # TODO: load from the hdf5 file?
        return self._xt

    @property
    def yt(self):
        # TODO: load from the hdf5 file
        return self._yt

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device
        B.set_global_device(device)

    @property
    def density_loc(self):
        return self._density_loc

    @density_loc.setter
    def density_loc(self, density_loc):
        density_loc = Path(density_loc)
        if self.load is True:
            if density_loc.exists():
                self._density_loc = density_loc
            else:
                raise FileNotFoundError(f"{density_loc} does not exist")
        elif density_loc.exists() and not self.overwrite:
            raise ValueError(f"{density_loc} already exists")
        else:
            with h5py.File(density_loc, "w") as f:
                f.create_group(Groups.MARGINAL_DENSITIES.value)
                f.create_group(Groups.TRAJECTORIES.value)
                f.attrs["num_functions_per_context_size"] = self.num_functions_per_context_size
                f.attrs["num_trajectories"] = self.num_trajectories
                f.attrs["context_sizes"] = self.context_sizes
                f.attrs["str(model)"] = str(self.model)
                f.attrs["str(data_generator)"] = str(self.data_generator)
                f.attrs["str(trajectory_generator)"] = str(self.trajectory_generator)
                if self.metadata is not None:
                    for k, v in self.metadata.items():
                        if k in f.attrs:
                            raise ValueError(f"{k} already exists in attributes")
                        else:
                            f.attrs[k] = v
        self._density_loc = density_loc

    @property
    def num_functions(self):
        return self._num_functions

    @property
    def context_sizes(self):
        return self._context_sizes

    @context_sizes.setter
    def context_sizes(self, context_sizes):
        self._context_sizes = np.array(context_sizes)
        self._num_functions = (
            self.num_functions_per_context_size * self.context_sizes.shape[0]
        )

    @cached_property
    def function_trajectory_sets(self):
        group_names = []
        with h5py.File(self.density_loc, "r") as f:
            for gn in f[Groups.TRAJECTORIES.value].keys():
                group_names.append(gn)
        for gn in group_names:
            ss = FunctionTrajectorySet.from_hdf5(
                hdf5_loc=self.density_loc, group_name=gn
            )
            self._function_trajectory_sets.append(ss)
        return self._function_trajectory_sets

    def generate_batch(self, batch_size=16, num_context=None):
        # This is messy since is changes the attributes of the generator.
        # Maybe fix at some point.
        self.data_generator.batch_size = batch_size
        if num_context is not None:
            # TODO: could also allow for ranges here?
            self.data_generator.num_context = UniformDiscrete(num_context, num_context)
        batch = self.data_generator.generate_batch()
        # choose batch with defined context? or use defined x_target?
        return batch

    def make_sample_sets(self):
        pbar = tqdm(enumerate(self.context_sizes), total=len(self.context_sizes))
        for ci, context_size in pbar:
            pbar.set_description(f"context set size: {context_size}")

            start_ind = ci * self.num_functions_per_context_size
            end_ind = (ci + 1) * self.num_functions_per_context_size
            # TODO: remove context_size from this and we get really good lls from AR?
            # Why? Only on a single point?
            batch = self.generate_batch(
                batch_size=self.num_functions_per_context_size, num_context=context_size
            )
            if isinstance(batch["xt"], AggregateInput):
                tmp_xt = batch["xt"].elements[0][0]  # only one output assumed
                tmp_yt = batch["yt"].elements[0]
                # I don't quite understand why it is indexed like this.
            else:
                tmp_xt = batch["xt"]
                tmp_yt = batch["yt"]
            self._xt[start_ind:end_ind, :, :] = tmp_xt
            self._yt[start_ind:end_ind, :, :] = tmp_yt
            contexts = batch["contexts"]

            # for fi in self.tqdm(range(self.num_functions_per_context_size)):
            # func_context = self.get_function_context(contexts, fi)
            func_context = contexts
            ss = FunctionTrajectorySet(
                hdf5_loc=self.density_loc,
                contexts=func_context,
                trajectory_generator=self.trajectory_generator,
                group_name=f"{context_size}",
            )
            ss.tqdm = self.tqdm
            ss.create_samples(self.model, self.num_trajectories)
        # Think about how to adapt this KL using ABC rejection sampling
        # Think about how to adapt to using chain rule of prob on joint

    def create_density_grid(
        self,
        density_eval="generated",
        density_kwargs=None,
        batch_size=100,
        # TODO: allow passing of group name to make separate densities from same
    ):
        if len(self.function_trajectory_sets) == 0:
            raise ValueError("No sample sets created yet!")
        with h5py.File(self.density_loc, "a") as f:
            md_grp = f[Groups.MARGINAL_DENSITIES.value]
            for ss in self.function_trajectory_sets:
                for func_ind in range(ss.num_functions):
                    md_grp.create_group(f"{ss.num_contexts}|{func_ind}")
        LOG.info(f"Creating density grid for {self.num_functions} sampled functions.")
        pbar = self.tqdm(
            enumerate(self.function_trajectory_sets),
            total=len(self.function_trajectory_sets),
        )
        for ss_ind, ss in pbar:
            self.inner_create_density_grid(
                ss,
                ss_ind,
                density_eval=density_eval,
                density_kwargs=density_kwargs,
                batch_size=batch_size,
            )

    def get_targets_and_density_eval_points(
        self, tfunc_ind, ss_ind, density_eval, density_kwargs
    ):
        # density_grid is not used anymore, but I'm leaving it because
        # I'm afraid of breaking something right now.
        # only the target point is used (which is the first col of the density eval points)
        func_ind = (ss_ind * self.num_functions_per_context_size) + tfunc_ind
        targets = self.xt[func_ind, :, :].reshape(-1, 1)
        if density_eval == "generated":
            yt0 = self.yt[func_ind, :, :]
            y_targets = yt0.reshape(-1, 1)
        elif density_eval == "grid":
            # TODO: add option which gets these values automatically by looking at some
            # aspect of underlying generator.
            yt0 = torch.arange(
                density_kwargs["start"], density_kwargs["end"], density_kwargs["step"]
            )
            # Add the real target to the start of the grid.
            true_target = self.yt[func_ind, :, :].reshape(-1, 1)
            y_targets = yt0.repeat(targets.shape[0], 1)
            y_targets = torch.cat([true_target, y_targets], dim=1)
        else:
            raise ValueError(f"density_eval must be one of 'generated' or 'grid'")
        self.num_density_eval_locations = y_targets.shape[1]
        targets = B.to_active_device(B.cast(FunctionTrajectorySet.dtype, targets))
        y_targets = B.to_active_device(B.cast(FunctionTrajectorySet.dtype, y_targets))
        return targets, y_targets

    def grid_loglikelihoods(self):
        grid = np.zeros((self.num_trajectories + 1, self.trajectory_length + 1))
        for nt in self.tqdm(range(self.num_trajectories + 1)):
            for tl in range(self.trajectory_length + 1):
                grid[nt, tl] = self.calc_loglikelihood(nt, tl)
        return grid

    def calc_loglikelihood(self, num_trajectories, trajectory_length):
        if num_trajectories > self.num_trajectories:
            raise ValueError(
                f"num_trajectories must be less than or equal to {self.num_trajectories}"
            )
        if trajectory_length > self.trajectory_length:
            raise ValueError(
                f"trajectory_length must be less than or equal to {self.trajectory_length}"
            )
        # TODO: use size from tensor here instead of the class attribute
        func_loglikelihoods = np.zeros((self.num_functions))
        for ss_ind, ss in enumerate(self.function_trajectory_sets):
            for func_ind in range(ss.num_functions):
                with h5py.File(self.density_loc, "r") as f:
                    gname = f"{ss.num_contexts}|{func_ind}"
                    grp = f[Groups.MARGINAL_DENSITIES.value][gname]
                    lh = grp[Datasets.LOG_LIKELIHOODS.value]
                    # the first of the eval points is the true y target
                    # that is why we evaluate the likelihood there
                    lh0 = lh[:, :num_trajectories, 0, trajectory_length]
                    expected_ll = get_func_expected_ll(lh0)

                    overall_func_ind = (
                        ss_ind * self.num_functions_per_context_size
                    ) + func_ind
                    func_loglikelihoods[overall_func_ind] = expected_ll
        return func_loglikelihoods.mean()

    def get_choices(self, num_contexts, func_ind, num_trajectories, trajectory_length):
        if num_contexts is None:
            num_contexts = np.random.choice(self.context_sizes)
        if num_contexts not in self.context_sizes:
            raise ValueError(f"num_contexts must be one of {self.context_sizes}")
        if func_ind is None:
            func_ind = np.random.choice(self.num_functions_per_context_size)
        if func_ind >= self.num_functions_per_context_size:
            raise ValueError(
                f"func_ind must be less than {self.num_functions_per_context_size}"
            )
        if num_trajectories is None:
            num_trajectories = np.random.choice(self.num_trajectories)
        if num_trajectories > self.num_trajectories:
            raise ValueError(
                f"num_trajectories must be less than or equal to {self.num_trajectories}"
            )
        if trajectory_length is None:
            trajectory_length = np.random.choice(self.trajectory_length)
        if trajectory_length > self.trajectory_length:
            raise ValueError(
                f"trajectory_length must be less than or equal to {self.trajectory_length}"
            )
        return num_contexts, func_ind, num_trajectories, trajectory_length

    def get_density(
        self,
        num_contexts=None,
        func_ind=None,
        num_trajectories=None,
        trajectory_length=None,
    ):
        num_contexts, func_ind, num_trajectories, trajectory_length = self.get_choices(
            num_contexts, func_ind, num_trajectories, trajectory_length
        )
        with h5py.File(self.density_loc, "a") as f:
            grp = f[Groups.MARGINAL_DENSITIES.value][
                f"{num_contexts}|{func_ind}"
            ]
            lh = grp[Datasets.LOG_LIKELIHOODS.value]
            density = lh[:, :num_trajectories, :, trajectory_length]
        return density

    def inner_create_density_grid(
        self,
        ss,
        ss_ind,
        density_eval="generated",
        density_kwargs=None,
        batch_size=100, # batch over target points to avoid memory issues
    ):
        xc_all_funcs, yc_all_funcs = append_contexts_to_samples(ss.contexts, ss.traj)
        outer_pbar = tqdm(np.arange(ss.num_functions))
        for func_ind in outer_pbar:
            outer_pbar.set_description(f"Context Index: {ss_ind}| Function Index: {func_ind}")
            xc_all = xc_all_funcs[:, func_ind, ...][:, None, ...]
            yc_all = yc_all_funcs[:, func_ind, ...][:, None, ...]
            targets, y_targets = self.get_targets_and_density_eval_points(
                func_ind, ss_ind, density_eval, density_kwargs
            )
            with h5py.File(self.density_loc, "a") as f:
                grp = f[Groups.MARGINAL_DENSITIES.value][
                    f"{ss.num_contexts}|{func_ind}"
                ]
                # Always the same number of targets throughout the batch.
                # Technically could add function draw as a dimension and add to tensor.
                grp.create_dataset(
                    Datasets.LOG_LIKELIHOODS.value,
                    (
                        self.num_targets,
                        self.num_trajectories,
                        # self.num_density_eval_locations,
                        1, # only write to target location
                        self.trajectory_length + 1,  # include trajectory length of 0.
                    ),
                    chunks=( # chunk same size with which we write the data
                        self.num_targets,
                        self.num_trajectories,
                        # self.num_density_eval_locations,
                        1,
                        1, # we write one trajectory length at a time, so store chunks like this
                    ),
                    compression="gzip",
                )
                mshape = (self.num_trajectories, self.num_targets, 1, 1, self.trajectory_length + 1)
                chunks = (self.num_trajectories, self.num_targets, 1, 1, 1)
                grp.create_dataset(
                    Datasets.MEANS.value, shape=mshape, chunks=chunks, compression="gzip")
                grp.create_dataset(
                    Datasets.VARIANCES.value, shape=mshape, chunks=chunks, compression="gzip")
                # append to existing if adding more points
                # make something to store index which tells which experiment it is from
                # write all config yaml as attributes?
                # write github hash repo as attribute?
                grp.create_dataset("y_density_evaluation_points", data=y_targets.cpu())
                grp.create_dataset("x_targets", data=targets.cpu())
                # (x_target locations)
                pbar = self.tqdm(
                    range(self.trajectory_length + 1),
                    total=self.trajectory_length + 1,
                    leave=False,
                )
                for tl_ind in pbar:
                    pbar.set_description(f"Trajectory length: {tl_ind}")
                    # i=0 -> no AR (b/c no trajectory, only context)
                    trunc_length = tl_ind + ss.num_contexts
                    xc, yc = truncate_samples(xc_all, yc_all, trunc_length)

                    ntarg = self.num_targets
                    ntraj = self.num_trajectories
                    all_lls = torch.Tensor(ntarg, ntraj, 1)
                    all_means = torch.Tensor(ntraj, ntarg, 1, 1)
                    all_vars = torch.Tensor(ntraj, ntarg, 1, 1)

                    xt = targets.reshape(-1, 1, 1)
                    xt = B.to_active_device(B.cast(FunctionTrajectorySet.dtype, xt))
                    xc = B.to_active_device(B.cast(FunctionTrajectorySet.dtype, xc))
                    yc = B.to_active_device(B.cast(FunctionTrajectorySet.dtype, yc))

                    # Batch the xts here to avoid CUDA memory issues.
                    for batch_inds in group_list(np.arange(xt.shape[0]), batch_size):
                        xt_batch = xt[batch_inds, ...]
                        xt_ag_batch = nps.AggregateInput((xt_batch, 0))
                        pred_batch = self.model(xc, yc, xt_ag_batch)
                        all_means[:, batch_inds, :, :] = pred_batch.mean.elements[0]
                        all_vars[:, batch_inds, :, :] = pred_batch.var.elements[0]

                        yt_batch = y_targets[batch_inds, 0].reshape(-1, 1)
                        yt_batch = B.to_active_device(B.cast(FunctionTrajectorySet.dtype, yt_batch))
                        yt_ll = pred_batch.logpdf(yt_batch)

                        lls = B.transpose(yt_ll).reshape(batch_size, ntraj)
                        all_lls[batch_inds, :, 0] = lls

                    all_means_np = all_means.cpu().detach().numpy()
                    all_vars_np = all_vars.cpu().detach().numpy()
                    all_llnp = all_lls.cpu().detach().numpy()
                    grp[Datasets.MEANS.value][..., tl_ind] = all_means_np
                    grp[Datasets.VARIANCES.value][..., tl_ind] = all_vars_np
                    grp[Datasets.LOG_LIKELIHOODS.value][:, :, :, tl_ind] = all_llnp


class FunctionTrajectorySet:

    dtype = torch.float32

    def __init__(
        self,
        hdf5_loc: Path,
        contexts=None,
        trajectory_generator=None,
        group_name=None,
    ):
        self.hdf5_loc = hdf5_loc
        self.data = None
        self.trajectory_generator = trajectory_generator
        self.tqdm = tqdm
        self.dtype = torch.float32
        self.group_name = group_name
        self.frozen = False

        go_init = False
        # TODO: this is quite convoluted. Probably can make more clear.
        if self.hdf5_loc.exists():
            with h5py.File(self.hdf5_loc, "r") as f:
                if group_name is not None:
                    if group_name in f[Groups.TRAJECTORIES.value].keys():
                        grp = f[Groups.TRAJECTORIES.value][group_name]
                    else:
                        go_init = True
                else:
                    grp = f

                if not go_init:
                    if "cx" in grp.keys():
                        LOG.info(f"{self.hdf5_loc}:{group_name} already created")
                        self.freeze()
                    else:
                        raise Exception(
                            f"{self.hdf5_loc}:{group_name} exists but cx not found."
                        )
        else:
            go_init = True

        if go_init:
            LOG.debug(f"{self.hdf5_loc}:{group_name} not found, creating")
            self.initialize_data(contexts)

    def initialize_data(self, contexts):
        # cx_np = contexts[0][0].cpu().detach().numpy().reshape(-1)
        # cy_np = contexts[0][1].cpu().detach().numpy().reshape(-1)
        if self.group_name is not None:
            mode = "a"
        else:
            mode = "w"
        with h5py.File(self.hdf5_loc, mode) as f:
            grp = self.get_group(f)
            grp.attrs["trajectory_generator"] = str(self.trajectory_generator)
            grp.attrs["sample_size"] = self.trajectory_generator.trajectory_length
            grp.create_dataset("cx", data=contexts[0][0].cpu())
            grp.create_dataset("cy", data=contexts[0][1].cpu())

    def freeze(self):
        self.frozen = True

    @classmethod
    def from_hdf5(cls, hdf5_loc, group_name=None):
        with h5py.File(hdf5_loc, "r") as f:
            if group_name is None:
                grp = f
            else:
                grp = f[Groups.TRAJECTORIES.value][group_name]
            cx = grp["cx"][:].reshape(1, 1, -1)
            cy = grp["cy"][:].reshape(1, 1, -1)
            contexts = cls._get_contexts(cx, cy)
            return cls(
                hdf5_loc=hdf5_loc,
                contexts=contexts,
                group_name=group_name,
            )

    def get_group(self, f):
        if self.group_name is not None:
            tj_grp = f[Groups.TRAJECTORIES.value]
            if self.group_name in tj_grp.keys():
                grp = tj_grp[self.group_name]
            else:
                grp = tj_grp.create_group(self.group_name)
        else:
            grp = f
        return grp

    @cached_property
    def sample_size(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            sample_size = int(grp.attrs["sample_size"])
        return sample_size

    @cached_property
    def contexts(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            cx = grp["cx"][:]
            cy = grp["cy"][:]
        fc = self._get_contexts(cx, cy)
        return fc

    @staticmethod
    def _get_contexts(cx, cy):
        cxt = B.to_active_device(B.cast(FunctionTrajectorySet.dtype, cx))
        cyt = B.to_active_device(B.cast(FunctionTrajectorySet.dtype, cy))
        fc = [(cxt, cyt)]
        return fc

    @cached_property
    def num_contexts(self):
        return self.contexts[0][0].shape[2]

    @cached_property
    def num_functions(self):
        return self.contexts[0][0].shape[0]

    def check_traj(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            if "y_traj" not in grp:
                raise Exception("Samples not yet created.")

    def get_traj(self, dset_name):
        self.check_traj()
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            dim_traj = grp[dset_name][:]
            # dim_traj = torch.Tensor(np_dim_traj).reshape(-1, 1, 1, self.sample_size)
        dim_traj = B.to_active_device(B.cast(self.dtype, dim_traj))
        return dim_traj

    @property
    def y_traj(self):
        y_traj = self.get_traj("y_traj")
        return y_traj

    @property
    def x_traj(self):
        x_traj = self.get_traj("x_traj")
        return x_traj

    @property
    def traj(self):
        self.check_traj()
        return [self.x_traj, self.y_traj]

    def create_samples(self, model, n_samples):
        if self.frozen is True:
            raise Exception("Samples already created.")
        with h5py.File(self.hdf5_loc, "a") as f:
            grp = self.get_group(f)
            if "y_traj" in grp:
                LOG.warning("Samples already created. Skipping.")
                return

        with h5py.File(self.hdf5_loc, "a") as f:
            grp = self.get_group(f)
            x_traj = grp.create_dataset(
                "x_traj",
                (n_samples, self.num_functions, 1, self.sample_size),
                dtype="float32",
            )
            y_traj = grp.create_dataset(
                "y_traj",
                (n_samples, self.num_functions, 1, self.sample_size),
                dtype="float32",
            )

            inner_samples = 1
            # TODO: could tweak this for more speed, but less stochasticity
            for i in self.tqdm(range(n_samples), leave=False):
                # TODO: assess whether this is fast enough, can I do it in parallel somehow?
                x_traj0 = self.trajectory_generator.generate(self.contexts[0][0].cpu())
                x_traj0 = B.to_active_device(B.cast(self.dtype, x_traj0))
                y_traj0 = torch.Tensor(x_traj0.shape)
                for j in range(x_traj0.shape[0]):
                    inner_shape = (1, x_traj0.shape[1], x_traj0.shape[2])
                    inner_context = get_function_context(self.contexts, j)
                    x_traj0_func0 = x_traj0[j, ...].reshape(inner_shape)
                    y_traj0_func0 = get_trajectories(
                        model, x_traj0_func0, inner_samples, inner_context)
                    y_traj0[j] = y_traj0_func0.reshape(inner_shape)
                x_traj[i, ...] = x_traj0.cpu().detach().numpy()
                y_traj[i, ...] = y_traj0.cpu().detach().numpy()
        self.freeze()


def get_trajectories(model, xi, n_mixtures, contexts):
    # Generate AR trajectories from the fixed selections in the domain.
    ag = nps.AggregateInput((xi, 0))
    mean, var, noiseless_samples, noisy_samples = nps.ar_predict(
        model, contexts, ag, num_samples=n_mixtures, order="given"
    )
    y_traj = noisy_samples.elements[0]#.reshape(xi.shape)
    # TODO: confirm that this is doing the right thing ^
    return y_traj


def load_model(weights, name, device="cpu"):
    print("Name:", name)
    if name in ["sawtooth", "synthetic-audio", "simple-mixture"]:
        model = get_model_gen()
    elif name in ["real-audio"]:
        model = get_model_real_audio()
    else:
        raise ValueError(f"Unknown model name: {name}")
    model.load_state_dict(torch.load(weights, map_location=device)["weights"])
    return model


def get_generator(generator_kwargs, num_context=None, specific_x=None, device="cpu"):
    ARGS = namedtuple("args", generator_kwargs)
    args = ARGS(**generator_kwargs)
    # General config. (copied from train.py)
    config = {
        "default": {
            "epochs": None,
            "rate": None,
            "also_ar": False,
        },
        "epsilon": 1e-8,
        "epsilon_start": 1e-2,
        "cholesky_retry_factor": 1e6,
        "fix_noise": None,
        "fix_noise_epochs": 3,
        "width": 256,
        "dim_embedding": 256,
        "enc_same": False,
        "num_heads": 8,
        "num_layers": 6,
        "unet_channels": (64,) * 6,
        "unet_strides": (1,) + (2,) * 5,
        "conv_channels": 64,
        "encoder_scales": None,
        "fullconvgnp_kernel_factor": 2,
        # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
        # doesn't make sense to set it to a value higher of the last hidden layer of
        # the CNN architecture. We therefore set it to 64.
        "num_basis_functions": 64,
    }

    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=2**6,
        num_tasks_cv=2**6,
        num_tasks_eval=2**6,
        device=device,
    )
    if isinstance(num_context, int):
        num_context = UniformDiscrete(num_context, num_context)
    elif num_context is None:
        num_context = UniformDiscrete(50, 50)
    # TODO: Should probably get generator more manually...
    # That way have more control of what is passed.
    # including control of batch size
    # also just grabbing first generator from eval lack clarity.
    gen = gens_eval()[0][1]
    # TODO: make number of targets an option
    gen.num_target = UniformDiscrete(500, 500)
    gen.num_context = num_context
    # has to be set after num_target b/c messy code with PhoneGenerator
    # When this is big, can run out of memory when making preds
    # of course, could remedy by batching, but don't feel like doing that right now
    # When its small, the resulting grid output animation is not as pretty.
    l = gen.dist_x_target.lowers[0]
    u = gen.dist_x_target.uppers[0]
    gen.dist_x_target = Grid(l, u)
    # This doesn't do anything in phone case
    # TODO: fixing this right now
    # REMOVE THIS, ONLY FOR GETTING A QUICK TEST OF VISUALS
    # gen.dist_x_context = UniformContinuous(0, 0)
    if specific_x is not None:
        if num_context.upper == 1:
            gen.dist_x_context = UniformContinuous(specific_x, specific_x)
        else:
            raise ValueError("Can't use specific x with many contexts")
    return gen


def get_model_gen():
    # Construct the model and load the weights.
    model = nps.construct_convgnp(  # where do I get these values? make sure they are correct
        dim_x=1,
        dim_y=1,
        unet_channels=(64,) * 6,
        points_per_unit=64,
        likelihood="het",
    )
    # model.load_state_dict(torch.load(weights, map_location="cpu")["weights"])
    return model


def get_model_real_audio():
    # Take these values from experiment/data/phone.py
    # TODO: This should be done in a more clear and automatable way
    num_channels = 7
    model = nps.construct_convgnp(
        points_per_unit=1,
        dim_x=1,
        dim_yc=(1,) * 1,
        dim_yt=1,
        likelihood="het",
        # conv_arch=args.arch,
        unet_channels=(128,) * num_channels,
        unet_strides=(1,) + (2,) * (num_channels - 1),
        conv_channels=64,
        conv_layers=6,
        conv_receptive_field=50,
        margin=0.25,  # 07_10
        # margin=1,  # 07_8
        encoder_scales=None,
        transform=None,
    )
    return model


def get_workers(workers):
    if workers == -1:
        return os.cpu_count()
    elif workers == "half":
        return int(os.cpu_count() / 2)
    elif workers is None:
        return 1
    else:
        return workers


def truncate_samples(xc_all, yc_all, max_len=None):
    # max_len here so that you can use a length less than the length of trajectory
    # used for the samples. Can be used to assess the impact of using different lengths
    # of trajectories.
    if max_len is not None:
        # LOG.info(f"Limiting trajectory length to {max_len}")
        xc = xc_all[..., :max_len]
        yc = yc_all[..., :max_len]
    else:
        xc = xc_all
        yc = yc_all
    return xc, yc


def generate_marginal_densities(
    out_marginal_densities, ss, model, targets, dxi, max_len=None, workers=None
):
    xc, yc = append_contexts_to_samples(ss.contexts, ss.traj)
    # TODO: iterate to full traj length and perform calc for all?
    # TODO: but reduce the number of x^(T) (targets) and only get density (dxi) for the
    # real value y^(T) (which we need to pass in to this function).
    xc, yc = truncate_samples(xc, yc, max_len)

    workers = get_workers(workers)
    with h5py.File(out_marginal_densities, "w") as f:
        # TODO: write out a reference to SampleSet
        # TODO: write out trajectory length truncation
        # Shape becomes max trajectory length so we have density for each traj length
        f.create_dataset("ad", (len(targets), len(dxi)), maxshape=(None, None))
        f.create_dataset("dxi", data=dxi.cpu())  # (grid_densities_calculated_on)
        f.create_dataset("targets", data=targets.cpu())  # (x_target locations)

        LOG.info(
            "{N(mu, var) = p_model(y_target|ar_context_i, x_target) "
            f"for y_target in targets (n={len(targets)}), "
            f"ar_context_i in trajectories (n={ss.y_traj.shape[0]}"
        )
        LOG.info("Generating predictive densities GMMs for all targets.")
        LOG.info("[[Sum(N(x=x0|mu, var)) for mu, var in preds] for x0 in dxi]")
        LOG.info(f"Using {workers} workers")

        dm = DensityMaker(xc, yc, model, dxi)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pbar = tqdm(executor.map(dm.generate, targets), total=len(targets))
            for i, ad0 in enumerate(pbar):
                f["ad"][i, :] = ad0

    return out_marginal_densities


class DensityMaker:
    def __init__(self, xc, yc, model):
        self.xc = xc
        self.yc = yc
        self.model = model

    def generate(self, xt_yt):
        xt, yt = xt_yt
        xt = xt.reshape(1, 1, -1)
        xt_ag = nps.AggregateInput((xt, 0))
        # TODO: Find out how to do this while setting the seed for reproducibility
        # Could do this in a batched fashion, I think
        pred = self.model(self.xc, self.yc, xt_ag)
        # TODO: for eval held out-loglik, only dxi of interest is the true value from
        # the observations.
        ad0 = plot_densities(pred, yt)
        return ad0


def append_contexts_to_samples(contexts, traj):
    # TODO: adapt to work with multiple outputs
    xc = traj[0]
    yc = traj[1]
    num_samples = traj[0].shape[0]
    repcx = contexts[0][0].repeat(num_samples, 1, 1, 1)
    repcy = contexts[0][1].repeat(num_samples, 1, 1, 1)
    xc = torch.cat((repcx, xc), axis=-1)
    yc = torch.cat((repcy, yc), axis=-1)

    return xc, yc


def plot_densities(pred, dxi):
    m2 = pred.mean.elements[0].cpu().detach().numpy().reshape(-1)
    v2 = pred.var.elements[0].cpu().detach().numpy().reshape(-1)

    # TODO: out put each component of the density and return for use in db
    densities = torch.zeros(len(m2), len(dxi))
    for i, (m, v) in enumerate(zip(m2, v2)):
        d0 = norm.pdf(dxi.cpu(), loc=m, scale=np.sqrt(v))
        densities[i, :] = torch.Tensor(d0)
    return densities


def get_device(device=None, gpu=None):
    if device:
        device = device
    elif gpu is not None:
        device = f"cuda:{gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def clean_config(config: dict) -> dict:
    # TODO: use defaultdict
    if "max_len" not in config:
        config["max_len"] = None
    if "num_samples" not in config:
        config["num_samples"] = None
    if "generator_kwargs" not in config:
        config["generator_kwargs"] = {}
    if "num_context" not in config:
        config["num_context"] = None
    if "specific_x" not in config:
        config["specific_x"] = None
    if "batch_size" not in config:
        config["batch_size"] = 100  # default batch size if missing
    config["model_weights"] = Path(config["model_weights"])
    return config


def make_heatmap(grd, config, outdir):
    grd[grd == -np.inf] = np.nan
    ax = sns.heatmap(grd)
    plt.xlabel("trajectory length")
    plt.ylabel("number of trajectories")
    gname = config["trajectory"]["generator"]
    plt.title(f"{config['name']} log-likelihoods with {gname} trajectory.")
    plt.savefig(outdir / f"heatmap.png")
    plt.clf()
    return outdir


def main(
    in_config: Path,
    out_model_dir: Path,
    device=None,
    gpu=None,
    exist_ok=False,
):
    with open(in_config, "r") as f0:
        orig_config = yaml.safe_load(f0)
    config = clean_config(orig_config)

    out_sampler_dir = out_model_dir / config["experiment"]

    LOG.info(f"Writing all results to \"{out_sampler_dir}\".")
    if out_sampler_dir.exists():
        if exist_ok:
            LOG.warning(f"{out_sampler_dir} already exists. Overwriting...")
        else:
            raise FileExistsError(f"{out_sampler_dir} already exists.")

    out_sampler_dir.mkdir(exist_ok=exist_ok)
    with open(out_sampler_dir / "config.yaml", "w") as f1:
        yaml.dump(orig_config, f1)

    device = get_device(device, gpu)
    B.set_global_device(device)

    data_generator = get_generator(
        config["generator_kwargs"],
        num_context=config["num_context"],
        specific_x=config["specific_x"],
        device=device,
    )
    trajectory_generator = construct_trajectory_gens(
        trajectory_length=config["trajectory"]["length"],
        x_range=(config["trajectory"]["low"], config["trajectory"]["high"]),
    )[config["trajectory"]["generator"]]
    # import ipdb; ipdb.set_trace()
    model = load_model(config["model_weights"], config["name"], device=device)
    model = model.to(device)

    density_loc = out_sampler_dir / "densities.hdf5"
    try:
        git_describe = subprocess.check_output(["git", "describe"]).strip()
    except:
        git_describe = "unknown"
    metadata = {"config": yaml.dump(config), "git_describe": git_describe}
    if "context_sizes" in config:
        context_sizes = np.array(config["context_sizes"])
    s = TrajectorySet(
        density_loc=density_loc,
        model=model,
        data_generator=data_generator,
        trajectory_generator=trajectory_generator,
        num_functions_per_context_size=config["num_functions"],
        num_trajectories=config["num_trajectories"],
        context_sizes=context_sizes,
        device=device,
        metadata=metadata,
    )
    LOG.info("Making Trajectories")
    s.make_sample_sets()
    LOG.info("Getting all loglikelihoods")
    s.create_density_grid(
        density_eval="grid",
        density_kwargs=config["density"]["range"],
        batch_size=config["batch_size"],
    )
    # grd = s.grid_loglikelihoods()
    # make_heatmap(grd, config, out_sampler_dir)
    # np.save(str(out_sampler_dir / "loglikelihoods_grid.npy"), grd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("--in_config", help="input yaml config file", type=Path)
    parser.add_argument("--out_sampler_dir", help="output directory", type=Path)
    parser.add_argument("--device", help="device to use", default=None)
    parser.add_argument("--gpu", help="gpu to use", type=int, default=None)
    parser.add_argument("--exist_ok", dest="exist_ok", action="store_true")
    parser.add_argument("--no-exist_ok", dest="exist_ok", action="store_false")
    parser.set_defaults(exist_ok=False)

    args = parser.parse_args()
    main(
        args.in_config,
        args.out_sampler_dir,
        args.device,
        args.gpu,
        args.exist_ok,
    )

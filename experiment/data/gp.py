from functools import partial
from itertools import product
import numpy as np

import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(name, args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    config["dim_x"] = args.dim_x
    config["dim_y"] = args.dim_y

    # Architecture choices specific for the GP experiments:
    # TODO: We should use a stride of 1 in the first layer, but for compatibility
    #    reasons with the models we already trained, we keep it like this.
    config["unet_strides"] = (2,) * 6
    config["conv_receptive_field"] = 4
    config["margin"] = 0.1
    
    if args.dim_x == 1:
        config["points_per_unit"] = 64
    elif args.dim_x == 2:
        # Reduce the PPU to reduce memory consumption.
        config["points_per_unit"] = 32
        # Since the PPU is reduced, we can also take off a layer of the UNet.
        config["unet_strides"] = config["unet_strides"][:-1]
        config["unet_channels"] = config["unet_channels"][:-1]
    else:
        raise RuntimeError(f"Invalid input dimensionality {args.dim_x}.")

    # Other settings specific to the GP experiments:
    config["plot"] = {
        1: {"range": (-2, 4), "axvline": [2]},
        2: {"range": ((-2, 2), (-2, 2))},
    }
    config["transform"] = None

    gen_train = nps.construct_predefined_gens(
        torch.float32,
        seed=1, # 10
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        pred_logpdf=False,
        pred_logpdf_diag=False,
        device=device,
        dp_epsilon_range=config["dp_epsilon_range"],
        dp_log10_delta_range=config["dp_log10_delta_range"],
        min_log10_scale=config["min_log10_scale"],
        max_log10_scale=config["max_log10_scale"],
        mean_diff=config["mean_diff"],
    )[name]

    gen_cv = lambda: nps.construct_predefined_gens(
        torch.float32,
        seed=20,  # Use a different seed!
        batch_size=args.batch_size,
        num_tasks=num_tasks_cv,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        device=device,
        dp_epsilon_range=config["dp_epsilon_range"],
        dp_log10_delta_range=config["dp_log10_delta_range"],
        min_log10_scale=config["min_log10_scale"],
        max_log10_scale=config["max_log10_scale"],
        mean_diff=config["mean_diff"],
    )[name]

    #def gens_eval():
    #    return [
    #        (
    #            eval_name,
    #            nps.construct_predefined_gens(
    #                torch.float32,
    #                seed=30,  # Use yet another seed!
    #                batch_size=args.batch_size,
    #                num_tasks=num_tasks_eval,
    #                dim_x=args.dim_x,
    #                dim_y=args.dim_y,
    #                pred_logpdf=True,
    #                pred_logpdf_diag=True,
    #                device=device,
    #                dp_epsilon_range=config["dp_epsilon_range"],
    #                dp_log10_delta_range=config["dp_log10_delta_range"],
    #                x_range_context=x_range_context,
    #                x_range_target=x_range_target,
    #                mean_diff=config["mean_diff"],
    #            )[args.data],
    #        )
    #        for eval_name, x_range_context, x_range_target in [
    #            ("interpolation in training range", (-2, 2), (-2, 2)),
    #            ("interpolation beyond training range", (2, 6), (2, 6)),
    #            ("extrapolation beyond training range", (-2, 2), (2, 6)),
    #        ]
    #    ]

    def gens_eval():
        return [
            (
                f"Using scale = {10**log_10_scale:.3f}, epsilon = {fixed_epsilon}",
                nps.construct_predefined_gens(
                    torch.float32,
                    seed=30,  # Use yet another seed!
                    batch_size=args.batch_size,
                    num_tasks=num_tasks_eval,
                    dim_x=args.dim_x,
                    dim_y=args.dim_y,
                    pred_logpdf=True,
                    pred_logpdf_diag=True,
                    device=device,
                    dp_epsilon_range=(fixed_epsilon, fixed_epsilon),
                    dp_log10_delta_range=config["dp_log10_delta_range"],
                    min_log10_scale=log_10_scale,
                    max_log10_scale=log_10_scale,
                )["scale-mix-eq"], # [args.data],
            )
            for fixed_epsilon, log_10_scale in product([9.], np.log10(np.array([0.10, 0.15, 0.20, 0.25, 0.35, 0.5, 1.0])))
        ]

    return gen_train, gen_cv, gens_eval

names = [
    "eq",
    "matern",
    "weakly-periodic",
    "mix-eq",
    "mix-matern",
    "scale-mix-eq",
    "scale-mix-matern",
    "mix-weakly-periodic",
    "sawtooth",
    "mixture",
]

for name in names:
    register_data(
        name,
        partial(setup, name),
        requires_dim_x=True,
        requires_dim_y=True,
    )

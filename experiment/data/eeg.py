import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):

    # Task dimensions: one input variable (time) and seven output variables (channels)
    config["dim_x"] = 1
    config["dim_y"] = 7

    # Architecture choices specific for the EEG experiments
    config["margin"] = 0.1
    config["transform"] = None
    config["dws_receptive_field"] = 1.
    config["unet_channels"] = (64,) * 6
    config["epsilon"] = 1e-6
    config["fullconvgnp_kernel_factor"] = 1
    config["points_per_unit"] = 256
    config["encoder_scales"] = 1 / 512

    # Other settings specific to the predator-prey experiments:
    config["plot"] = {1: {"range": (0, 1), "axvline": []}}

    gen_train = nps.EEGGenerator(
        dtype=torch.float32,
        split_seed=10,
        split="train",
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        device=device,
    )

    gen_cv = lambda: (
        nps.EEGGenerator(
            dtype=torch.float32,
            split_seed=20,
            split="valid",
            batch_size=args.batch_size,
            num_tasks=num_tasks_cv,
            device=device,
        )
    )

    def gens_eval():

        gen_eval_scattered = nps.EEGGenerator(
            dtype=torch.float32,
            split_seed=30,
            split="test",
            batch_size=args.batch_size,
            num_tasks=num_tasks_eval,
            single_channel=False,
            device=device,
        )

        gen_eval_single_channel = nps.EEGGenerator(
            dtype=torch.float32,
            split_seed=30,
            split="test",
            batch_size=args.batch_size,
            num_tasks=num_tasks_eval,
            single_channel=True,
            device=device,
        )

        gens = [
            #("standard eval", gen_eval_scattered),
            ("single channel eval", gen_eval_single_channel),
        ]

        return gens # [("standard eval", gen_eval_scattered)]

    return gen_train, gen_cv, gens_eval




register_data("eeg", setup)

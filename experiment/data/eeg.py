import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    # Task dimensions: one input variable (time) and seven output variables (channels)
    config["dim_x"] = 1
    config["dim_y"] = 7
    config["rate"] = 1e-4
    config["epochs"] = 200

    # Architecture choices specific for the EEG experiments:
    config["transform"] = None
    config["epsilon"] = 1e-6
    config["enc_same"] = True

    # Configure the convolutional models:
    config["points_per_unit"] = 256
    config["margin"] = 0.1
    config["conv_receptive_field"] = 1.0
    config["unet_strides"] = (1,) + (2,) * 6
    config["unet_channels"] = (64,) * 7
    config["fullconvgnp_kernel_factor"] = 1

    # Other settings specific to the EEG experiments:
    config["plot"] = {1: {"range": (0, 1), "axvline": []}}

    gen_train = nps.EEGGenerator(
        dtype=torch.float32,
        seed=0,
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        subset="train",
        device=device,
    )

    gen_cv = lambda: (
        nps.EEGGenerator(
            dtype=torch.float32,
            seed=20,
            batch_size=args.batch_size,
            num_tasks=num_tasks_cv,
            subset="cv",
            device=device,
        )
    )

    def gens_eval():
        return [
            (
                "Interpolation",
                nps.EEGGenerator(
                    dtype=torch.float32,
                    batch_size=args.batch_size,
                    num_tasks=num_tasks_eval,
                    mode="interpolation",
                    subset="eval",
                    device=device,
                ),
            ),
            (
                "Forecasting",
                nps.EEGGenerator(
                    dtype=torch.float32,
                    batch_size=args.batch_size,
                    num_tasks=num_tasks_eval,
                    mode="forecasting",
                    subset="eval",
                    device=device,
                ),
            ),
            (
                "Reconstruction",
                nps.EEGGenerator(
                    dtype=torch.float32,
                    batch_size=args.batch_size,
                    num_tasks=num_tasks_eval,
                    mode="reconstruction",
                    subset="eval",
                    device=device,
                ),
            ),
        ]

    return gen_train, gen_cv, gens_eval


register_data("eeg", setup)

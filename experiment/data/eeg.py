import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    config["default"]["rate"] = 5e-5 # 2e-4
    config["default"]["epochs"] = 1000 # 200
    config["dim_x"] = 1
    config["dim_y"] = 7

    # Architecture choices specific for the EEG experiments:
    config["transform"] = None
    config["epsilon"] = 1e-6
    config["enc_same"] = True

    # Configure the convolutional models:
    config["points_per_unit"] = 256
    config["margin"] = 0.1
    config["conv_receptive_field"] = 1.0
    config["unet_strides"] = (1,) + (2,) * 5
    
    # Increase the capacity of the ConvCNP, ConvGNP, and ConvNP to account for the many
    # outputs. The FullConvGNP is already large enough...
    if args.model in {"convcnp", "convgnp"}:
        config["unet_channels"] = (128,) * 6
    elif args.model == "convnp":
        config["unet_channels"] = (96,) * 6
    else:
        config["unet_channels"] = (64,) * 6
    config["encoder_scales"] = 0.77 / 256
    config["fullconvgnp_kernel_factor"] = 1

    # Other settings specific to the EEG experiments:
    config["plot"] = {1: {"range": (0, 1), "axvline": []}}

    gen_train = nps.EEGGenerator(
        dtype=torch.float32,
        seed=0,
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        mode=config["eeg_mode"], # "random",
        subset="train",
        device=device,
    )

    gen_cv = lambda: (
        nps.EEGGenerator(
            dtype=torch.float32,
            seed=20,
            batch_size=args.batch_size,
            num_tasks=num_tasks_cv,
            mode=config["eeg_mode"], # "random",
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

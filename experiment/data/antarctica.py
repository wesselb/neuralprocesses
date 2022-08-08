import os
import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    
    root_dir = f"{os.getcwd()}/antarctica-data"
    
    config["default"]["rate"] = 1e-4
    config["default"]["epochs"] = 200
    config["dim_x"] = 2
    config["dim_y"] = 2
    
    num_tasks_train = 10**4
    num_tasks_cv = 10**3
    num_tasks_eval = 10**3

    # Configure the convolutional models:
    config["points_per_unit"] = 256
    config["margin"] = 0.2
    config["conv_receptive_field"] = 1.
    config["unet_strides"] = (1,) + (2,) * 5
    
    config["unet_channels"] = (64, 64, 64, 64, 64, 64)
    config["encoder_scales"] = 2 / config["points_per_unit"]
    config["transform"] = None

    # Other settings specific to the EEG experiments:
    config["plot"] = {1: {"range": (0, 1), "axvline": []}}

    gen_train = nps.AntarcticaGenerator(
        root_dir=root_dir,
        dtype=torch.float32,
        seed=0,
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        subset="train",
        device=device,
    )

    gen_cv = lambda: (
        nps.AntarcticaGenerator(
            root_dir=root_dir,
            dtype=torch.float32,
            seed=0,
            batch_size=args.batch_size,
            num_tasks=num_tasks_cv,
            device=device,
        )
    )
    
    def gens_eval():
        return []

    return gen_train, gen_cv, gens_eval

register_data("antarctica", setup)

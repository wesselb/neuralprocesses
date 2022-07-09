from functools import partial

import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(data_task, args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    config["default"]["rate"] = 2.5e-5
    config["default"]["epochs"] = 500
    config["dim_x"] = 1
    config["dim_y"] = 1

    # Architecture choices specific for the phone experiments:
    config["transform"] = None

    # Configure the convolutional models:
    num_channels = 9
    config["points_per_unit"] = 4
    config["margin"] = 0.25
    # config["unet_kernels"] = 8 <- this does not get passed to convcnp
    config["conv_receptive_field"] = None  # not needed for unet architecture
    config["unet_strides"] = (1,) + (2,) * (num_channels-1)
    config["unet_channels"] = (128,) * num_channels  # Increase Capacity

    # Other settings specific to the phone experiments:
    config["plot"] = {
        1: {"range": (0, 800), "axvline": []},
        2: {"range": (0, 1600), "axvline": [800]},
    }

    gen_train = nps.PhoneGenerator(
        torch.float32,
        seed=10,
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        subset="train",
        # mode="random", # Things seem to go very poorly when use random...
        mode="interpolation",
        device=device,
        data_task=data_task,
    )
    gen_cv = lambda: nps.PhoneGenerator(
        torch.float32,
        seed=20,
        batch_size=args.batch_size,
        num_tasks=num_tasks_cv,
        subset="cv",
        # mode="random",
        mode="interpolation",
        device=device,
        data_task=data_task,
    )

    def gens_eval():
        return [
            (
                "Interpolation",
                nps.PhoneGenerator(
                    torch.float32,
                    seed=30,
                    num_tasks=num_tasks_eval // args.batch_size,
                    mode="interpolation",
                    subset="eval",
                    device=device,
                    data_task=data_task,
                ),
            ),
            (
                "Forecasting",
                nps.PhoneGenerator(
                    torch.float32,
                    seed=30,
                    num_tasks=num_tasks_eval // args.batch_size,
                    mode="forecasting",
                    device=device,
                    subset="eval",
                    data_task=data_task,
                ),
            ),
            (
                "Reconstruction",
                nps.PhoneGenerator(
                    torch.float32,
                    seed=30,
                    num_tasks=num_tasks_eval // args.batch_size,
                    mode="reconstruction",
                    device=device,
                    subset="eval",
                    data_task=data_task,
                ),
            ),
        ]

    return gen_train, gen_cv, gens_eval


# TODO: register more data types (more phones)
register_data("phone", partial(setup, ("iy",)))

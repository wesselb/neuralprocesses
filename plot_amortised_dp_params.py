import argparse
import os
import sys
import time
import warnings
from functools import partial

import experiment as exp
import lab as B
import neuralprocesses.torch as nps
import numpy as np
import torch
import wbml.out as out
from matrix.util import ToDenseWarning
from wbml.experiment import WorkingDirectory
from neuralprocesses.coders.setconv.privacy_accounting import find_sens_per_sigma

import matplotlib.pyplot as plt

__all__ = ["main"]

warnings.filterwarnings("ignore", category=ToDenseWarning)

def main(**kw_args):

    # Setup arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="*", default=["_experiments"])
    parser.add_argument("--subdir", type=str, nargs="*")
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--dim-x", type=int, default=1)
    parser.add_argument("--dim-y", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--arch",
        choices=[
            "unet",
            "unet-sep",
            "unet-res",
            "unet-res-sep",
            "conv",
            "conv-sep",
            "conv-res",
            "conv-res-sep",
        ],
        default="unet",
    )
    parser.add_argument(
        "--data",
        choices=exp.data,
        default="eq",
    )
    parser.add_argument("--mean-diff", type=float, default=None)
    parser.add_argument("--objective", choices=["loglik", "elbo"], default="loglik")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--resume-at-epoch", type=int)
    parser.add_argument("--train-fast", action="store_true")
    parser.add_argument("--check-completed", action="store_true")
    parser.add_argument("--unnormalised", action="store_true")
    
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--ar", action="store_true")
    parser.add_argument("--also-ar", action="store_true")
    parser.add_argument("--no-ar", action="store_true")
    parser.add_argument("--experiment-setting", type=str, nargs="*")
    parser.add_argument("--encoder-scales", type=float, default=None)

    parser.add_argument("--patch", type=str)
    parser.add_argument("--min-log10-scale", type=float, default=np.log10(0.1))
    parser.add_argument("--max-log10-scale", type=float, default=np.log10(5.0))

    parser.add_argument("--dp-epsilon-min", type=float, default=1.)
    parser.add_argument("--dp-epsilon-max", type=float, default=10.)
    parser.add_argument("--dp-log10-delta-min", type=float, default=-3.)
    parser.add_argument("--dp-log10-delta-max", type=float, default=-3.)
    parser.add_argument("--dp-y-bound", type=float, default=2.)
    parser.add_argument("--dp-use-noise-channels", default=False, action="store_true")
    parser.add_argument("--dp-amortise-params", default=False, action="store_true")

    if kw_args:
        # Load the arguments from the keyword arguments passed to the function.
        # Carefully convert these to command line arguments.
        args = parser.parse_args(
            sum(
                [
                    ["--" + k.replace("_", "-")] + ([str(v)] if v is not True else [])
                    for k, v in kw_args.items()
                ],
                [],
            )
        )
    else:
        args = parser.parse_args()

    # Remove the dimensionality specification if the experiment doesn't need it.
    if not exp.data[args.data]["requires_dim_x"]:
        del args.dim_x
    if not exp.data[args.data]["requires_dim_y"]:
        del args.dim_y

    # Ensure that `args.experiment_setting` is always a list.
    if not args.experiment_setting:
        args.experiment_setting = []

    # Determine settings for the setup of the script.
    suffix = ""
    observe = False
    if args.check_completed or args.no_action or args.load:
        observe = True
        
    else:
        suffix = "_evaluate"
        if args.ar:
            suffix += "_ar"

    dp_log10_delta_range = (args.dp_log10_delta_min, args.dp_log10_delta_max)
    
    def get_model_name(amortisation, epsilon=None):
        
        assert epsilon in [None, 1, 3, 9]
        
        eps_min = 1 if epsilon is None else epsilon
        eps_max = 10 if epsilon is None else epsilon
        
        model_name = "dpconvcnp_x"
        model_name = model_name + ("x" if not amortisation else "a")
        model_name = model_name + "_"
        model_name = model_name + f"{eps_min:.0f}-{eps_max:.0f}_"
        model_name = model_name + f"{dp_log10_delta_range[0]:.0f}-{dp_log10_delta_range[1]:.0f}"
        
        return model_name
        
    def get_wd(model_name):
        
        wd = WorkingDirectory(
            *args.root,
            *(args.subdir or ()),
            args.data,
            *((f"x{args.dim_x}_y{args.dim_y}",) if hasattr(args, "dim_x") else ()),
            model_name,
            *((args.arch,) if hasattr(args, "arch") else ()),
            args.objective,
            log=f"log{suffix}.txt",
            diff=f"diff{suffix}.txt",
            observe=observe,
        )
        
        return wd
    
    def make_model(config, amortisation):
        
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            unet_strides=config["unet_strides"],
            conv_channels=config["conv_channels"],
            conv_layers=config["num_layers"],
            conv_receptive_field=config["conv_receptive_field"],
            margin=config["margin"],
            encoder_scales=config["encoder_scales"],
            transform=config["transform"],
            divide_by_density=False,
            use_dp=True,
            amortise_dp_params=amortisation,
            use_dp_noise_channels=args.dp_use_noise_channels,
            dp_y_bound=args.dp_y_bound,
        )
        
        return model

    f_model_name = get_model_name(amortisation=False)
    f1_model_name = get_model_name(amortisation=False, epsilon=1)
    f3_model_name = get_model_name(amortisation=False, epsilon=3)
    f9_model_name = get_model_name(amortisation=False, epsilon=9)
    a_model_name = get_model_name(amortisation=True)
    
    # Setup script.
    if not observe:
        out.report_time = True
        
    fwd = get_wd(f_model_name)
    f1wd = get_wd(f1_model_name)
    f3wd = get_wd(f3_model_name)
    f9wd = get_wd(f9_model_name)
    awd = get_wd(a_model_name)

    # Determine which device to use. Try to use a GPU if one is available.
    if args.device:
        device = args.device
    elif args.gpu is not None:
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    B.set_global_device(device)
    # Maintain an explicit random state through the execution.
    state = B.create_random_state(torch.float32, seed=0)

    # General config.
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
        "encoder_scales": None or args.encoder_scales,
        "fullconvgnp_kernel_factor": 2,
        "mean_diff": args.mean_diff,
        # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
        # doesn't make sense to set it to a value higher of the last hidden layer of
        # the CNN architecture. We therefore set it to 64.
        "num_basis_functions": 64,
        "use_dp_noise_channels": args.dp_use_noise_channels,
        "dp_epsilon_range": (1., 1.),
        "dp_log10_delta_range": dp_log10_delta_range,
        "min_log10_scale": args.min_log10_scale,
        "max_log10_scale": args.max_log10_scale,
    }

    # Setup config with side effects
    _ = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=1,
        num_tasks_cv=1,
        num_tasks_eval=1,
        device=device,
    )

    f_model = make_model(config, amortisation=False)
    f1_model = make_model(config, amortisation=False)
    f3_model = make_model(config, amortisation=False)
    f9_model = make_model(config, amortisation=False)
    a_model = make_model(config, amortisation=True)

    # Ensure that the f_model is on the GPU and print the setup.
    f_model = f_model.to(device)

    f_model.load_state_dict(
        torch.load(fwd.file("model-best.torch"), map_location=device)["weights"]
    )

    # Ensure that the f_model is on the GPU and print the setup.
    f1_model = f1_model.to(device)

    f1_model.load_state_dict(
        torch.load(f1wd.file("model-best.torch"), map_location=device)["weights"]
    )

#     # Ensure that the f_model is on the GPU and print the setup.
#     f3_model = f3_model.to(device)

#     f3_model.load_state_dict(
#         torch.load(f3wd.file("model-best.torch"), map_location=device)["weights"]
#     )

    # Ensure that the f_model is on the GPU and print the setup.
    f9_model = f9_model.to(device)

    f9_model.load_state_dict(
        torch.load(f9wd.file("model-best.torch"), map_location=device)["weights"]
    )

    # Ensure that the a_model is on the GPU and print the setup.
    a_model = a_model.to(device)

    a_model.load_state_dict(
        torch.load(awd.file("model-best.torch"), map_location=device)["weights"]
    )
    
    def get_t_y_sigmas(model, sens_per_sigma, amortised):
    
        t = model.encoder.coder[1][0].t(sens_per_sigma).detach().cpu().numpy()
        y_bound = model.encoder.coder[1][0].y_bound(sens_per_sigma).detach().cpu().numpy()
        value_sigma = model.encoder.coder[1][0].value_sigma(sens_per_sigma).detach().cpu().numpy()
        density_sigma = model.encoder.coder[1][0].density_sigma(sens_per_sigma).detach().cpu().numpy()
            
        return t, y_bound, value_sigma, density_sigma
    
    sens_per_sigma = torch.linspace(2e-1, 2.5, 100).to(device)
    
    f1_sens_per_sigma = find_sens_per_sigma(epsilon=1., delta_bound=0.001)
    f9_sens_per_sigma = find_sens_per_sigma(epsilon=9., delta_bound=0.001)
        
    f_t, f_y_bound, f_value_sigma, f_density_sigma = get_t_y_sigmas(f_model, sens_per_sigma, amortised=False)
    f1_t, f1_y_bound, f1_value_sigma, f1_density_sigma = get_t_y_sigmas(f1_model, torch.tensor(f1_sens_per_sigma).to(device), amortised=False)
    # f3_t, f3_y_bound, f3_value_sigma, f3_density_sigma = get_t_y_sigmas(f3_model, sens_per_sigma, amortised=False)
    f9_t, f9_y_bound, f9_value_sigma, f9_density_sigma = get_t_y_sigmas(f9_model, torch.tensor(f9_sens_per_sigma).to(device), amortised=False)
    a_t, a_y_bound, a_value_sigma, a_density_sigma = get_t_y_sigmas(a_model, sens_per_sigma, amortised=True)
    
    sens_per_sigma = sens_per_sigma.detach().cpu().numpy()
        
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(sens_per_sigma, f_t * np.ones_like(sens_per_sigma), color="tab:red")
    plt.scatter([f1_sens_per_sigma], [f1_t], color="tab:green", marker="x", zorder=10)
    plt.scatter([f9_sens_per_sigma], [f9_t], color="tab:purple", marker="x", zorder=10)
    plt.plot(sens_per_sigma, a_t, color="tab:blue")
    plt.ylim([0., 1.])
    plt.title("Noise weight $t$")
    plt.xlabel("Sensitivity per $\\sigma$")
    
    plt.subplot(1, 3, 2)
    plt.plot(sens_per_sigma, f_y_bound * np.ones_like(sens_per_sigma), color="tab:red")
    plt.scatter([f1_sens_per_sigma], [f1_y_bound], color="tab:green", marker="x", zorder=10)
    plt.scatter([f9_sens_per_sigma], [f9_y_bound], color="tab:purple", marker="x", zorder=10)
    plt.plot(sens_per_sigma, a_y_bound, color="tab:blue")
    plt.ylim([1., 2.5])
    plt.title("Clipping threshold $y_b$")
    plt.xlabel("Sensitivity per $\\sigma$")
    
    plt.subplot(1, 3, 3)
    plt.plot(sens_per_sigma, f_value_sigma, color="tab:red", label="Value $\\sigma$")
    plt.plot(sens_per_sigma, f_density_sigma, "--", color="tab:red", label="Density $\\sigma$")
    plt.scatter([f1_sens_per_sigma], [f1_value_sigma], marker="o", color="tab:green", zorder=5, linewidth=1., edgecolor="k")
    plt.scatter([f9_sens_per_sigma], [f9_value_sigma], marker="o", color="tab:purple", zorder=5, linewidth=1., edgecolor="k")
    plt.plot(sens_per_sigma, a_value_sigma, color="tab:blue", label="Value $\\sigma$")
    plt.plot(sens_per_sigma, a_density_sigma, "--", color="tab:blue", label="Density $\\sigma$")
    plt.scatter([f1_sens_per_sigma], [f1_density_sigma], marker="o", color="white", linewidth=1., edgecolor="tab:green", zorder=5)
    plt.scatter([f9_sens_per_sigma], [f9_density_sigma], marker="o", color="white", linewidth=1., edgecolor="tab:purple", zorder=5)
    plt.legend(fontsize=12)
    plt.yscale("log")
    plt.title("Density and value $\\sigma$")
    plt.xlabel("Sensitivity per $\\sigma$")
    
    plt.tight_layout()
    plt.savefig("amortised.png")


if __name__ == "__main__":
    main()

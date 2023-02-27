import argparse
import os
import sys
import time
import warnings
from functools import partial

import json
import experiment as exp
import lab as B
import neuralprocesses.torch as nps
import dpsgp
import numpy as np
import torch
import wbml.out as out
import stheno
import optuna
import matplotlib.pyplot as plt
from matrix.util import ToDenseWarning
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

__all__ = ["main"]

warnings.filterwarnings("ignore", category=ToDenseWarning)


def train_model(model, optimiser, epochs, train_loader, trial=None):
    xc, yc = train_loader.dataset.tensors
    epochs_iter = tqdm(range(epochs), desc="Epoch")
    for epoch in epochs_iter:
        model.train()
        # Within each iteration, we will go over each minibatch of data
        for x_batch, y_batch in train_loader:
            if len(x_batch) == 0:
                continue

            optimiser.zero_grad()

            qf_params = model(x_batch)
            qf_loc = qf_params[:, 0]
            qf_cov = qf_params[:, 1]
            qf = torch.distributions.Normal(qf_loc, qf_cov.pow(0.5))

            exp_ll = model.likelihood.expected_log_prob(y_batch, qf).sum() * (
                len(xc) / len(x_batch)
            )

            kl = model._module.kl_divergence()

            elbo = exp_ll - kl
            (-elbo).backward()

            optimiser.step()

            metrics = {
                "elbo": elbo.item(),
            }
            epochs_iter.set_postfix(metrics)

        model.eval()
        with torch.no_grad():
            qf_params = model(xc)
            qf_loc = qf_params[:, 0]
            qf_cov = qf_params[:, 1]
            qf = torch.distributions.Normal(qf_loc, qf_cov.pow(0.5))

            exp_ll = model.likelihood.expected_log_prob(yc, qf).sum()
            kl = model._module.kl_divergence()
            elbo = (exp_ll - kl).item()

    return elbo


def dp_train_model(xc, yc, train_args, trial=None):
    kernel = dpsgp.kernels.RBFKernel()
    init_z = torch.linspace(xc.min(), xc.max(), train_args.num_inducing).unsqueeze(-1)

    # Initialise using true lengthscale and noise values. TODO: use BayesOpt on initialisation.
    kernel.lengthscale = 0.25
    likelihood = dpsgp.likelihoods.GaussianLikelihood(noise=0.05)
    model = dpsgp.sgp.SparseGP(kernel, likelihood, init_z)

    if train_args.use_true_kernel:
        kernel.lengthscale = 0.25
        kernel.log_lengthscale.requires_grad = False
        kernel.scale = 1.0
        kernel.log_scale.requires_grad = False
        likelihood.noise = 0.05
        likelihood.log_noise.requires_grad = False

    train_dataset = TensorDataset(xc, yc)
    train_loader = DataLoader(
        train_dataset, batch_size=train_args.batch_size, shuffle=True
    )

    privacy_engine = PrivacyEngine(accountant="rdp")
    optimiser = torch.optim.Adam(model.parameters(), lr=train_args.lr)

    model, optimiser, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimiser,
        data_loader=train_loader,
        epochs=train_args.epochs,
        target_epsilon=train_args.epsilon,
        target_delta=train_args.delta,
        max_grad_norm=train_args.max_grad_norm,
        grad_sample_mode="functorch",
    )
    return train_model(model, optimiser, train_args.epochs, train_loader, trial)


def batch_dp_train_model(xc, yc, train_args, trial=None):
    elbo = 0.0
    for xc_batch, yc_batch in zip(xc, yc):
        elbo += dp_train_model(xc_batch, yc_batch, train_args, trial)

    return elbo


# Define optuna objective function to be maximised.
def objective(epsilon, delta, xc, yc, use_true_kernel, trial):
    # Suggest values of hyperparameters using a trial object.
    num_inducing = trial.suggest_int("num_inducing", 10, 20)
    epochs = trial.suggest_int("epochs", 50, 300)
    batch_size = trial.suggest_int("batch_size", 10, 100)
    lr = trial.suggest_float("lr", 1e-3, 5e-2, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 1e-1, 1e1, log=True)

    train_args = {
        "epsilon": epsilon,
        "delta": delta,
        "num_inducing": num_inducing,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_grad_norm": max_grad_norm,
        "use_true_kernel": use_true_kernel,
    }
    train_args = argparse.Namespace(**train_args)

    elbo = batch_dp_train_model(xc, yc, train_args, trial)

    return elbo


def visualise_1d(model, gen, batch, path, config, epsilon, delta, best_trial, elbo):
    try:
        plot_config = config["plot"][1]
    except KeyError:
        return

    # Define points to predict at.
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), *plot_config["range"], 200).unsqueeze(-1)

    # Predict with model.
    with torch.no_grad():
        qf_params = model(x)
        mean = qf_params[:, 0].unsqueeze(-1)
        qf_var = qf_params[:, 1].unsqueeze(-1)
        var = qf_var + model.likelihood.noise

    plt.figure(figsize=(8, 6 * config["dim_y"]))

    for i in range(config["dim_y"]):
        plt.subplot(config["dim_y"], 1, 1 + i)

        # Plot context and target.
        plt.scatter(
            nps.batch_xc(batch, i)[0, 0],
            nps.batch_yc(batch, i)[0],
            label="Context",
            style="train",
            s=20,
        )

        plt.scatter(
            nps.batch_xt(batch, i)[0, 0],
            nps.batch_yt(batch, i)[0],
            label="Target",
            style="test",
            s=20,
        )

        # Plot prediction.
        err = 1.96 * B.sqrt(var[:, i])
        plt.plot(
            x,
            mean[:, i],
            label="Prediction",
            style="pred",
        )
        plt.fill_between(
            x,
            mean[:, i] - err,
            mean[:, i] + err,
            style="pred",
        )

        # Plot prediction by ground truth.
        if (hasattr(gen, "kernel") or hasattr(gen, "kernel_type")) and config[
            "dim_y"
        ] == 1:
            if hasattr(gen, "kernel_type"):
                f = stheno.GP(gen.kernel_type().stretch(batch["scale"]))

            else:
                f = stheno.GP(gen.kernel)

            # Make sure that everything is of `float64`s and on the GPU.
            noise = B.to_active_device(B.cast(torch.float64, gen.noise))
            xc = B.cast(torch.float64, nps.batch_xc(batch, 0)[0, 0])
            yc = B.cast(torch.float64, nps.batch_yc(batch, 0)[0])
            x = B.cast(torch.float64, x)
            # Compute posterior GP.
            f_post = f | (f(xc, noise), yc)
            mean, lower, upper = f_post(x).marginal_credible_bounds()

            lower = mean - 1.96 * (((mean - lower) / 1.96) ** 2.0 + noise) ** 0.5
            upper = mean + 1.96 * (((upper - mean) / 1.96) ** 2.0 + noise) ** 0.5

            plt.plot(x, mean, label="Truth", style="pred2")
            plt.plot(x, lower, style="pred2")
            plt.plot(x, upper, style="pred2")

        for x_axvline in plot_config["axvline"]:
            plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
            nps.batch_yt(batch, i)[0]

        N = nps.batch_yc(batch, i)[0].shape[0]
        ell = 0.25 if "scale" not in batch else batch["scale"].detach().cpu().numpy()[0]

        plt.gca().set_title(
            f"$N = {N:.0f}$  "
            + f"$\\ell = {ell:.3f}$  "
            + f"$N\\ell \\approx {N*ell:.0f}$  "
            + f"$\\epsilon = {epsilon:.2f}$  "
            + f"$\\delta = {delta:.3f}$  "
            + f"$({best_trial:.1f}, {elbo:.1f})$",
            fontsize=18,
        )

        plt.xlim(B.min(x), B.max(x))
        tweak()

    plt.savefig(path)
    plt.close()


def evaluate(model, xt: torch.Tensor, yt: torch.Tensor):
    # Predict with model.
    with torch.no_grad():
        qf_params = model(xt)
        mean = qf_params[:, 0].unsqueeze(-1)
        var = qf_params[:, 1].unsqueeze(-1) + model.likelihood.noise

    qf = torch.distributions.Normal(mean, var.pow(0.5))

    # Compute some predictive metrics.
    metrics = {
        "mean_loglik": qf.log_prob(yt).mean().item(),
        "rmse": (qf.loc - yt).pow(2).mean().sqrt().item(),
    }
    return metrics


def main(**kw_args):
    # Setup arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="*", default=["_experiments"])
    parser.add_argument("--subdir", type=str, nargs="*")
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--dim-x", type=int, default=1)
    parser.add_argument("--dim-y", type=int, default=1)
    parser.add_argument(
        "--data",
        choices=exp.data,
        default="eq",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mean-diff", type=float, default=None)
    parser.add_argument("--train-fast", action="store_true")
    parser.add_argument("--evaluate-fast", action="store_true")
    parser.add_argument("--evaluate-num-plots", type=int, default=5)

    parser.add_argument("--min-log10-scale", type=float, default=np.log10(0.1))
    parser.add_argument("--max-log10-scale", type=float, default=np.log10(5.0))

    parser.add_argument("--dp-epsilon-min", type=float, default=1.0)
    parser.add_argument("--dp-epsilon-max", type=float, default=9.0)
    parser.add_argument("--dp-log10-delta-min", type=float, default=-3.0)
    parser.add_argument("--dp-log10-delta-max", type=float, default=-3.0)

    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--meta-train-hypers", action="store_true", default=False)
    parser.add_argument("--meta-train-size", type=int, default=1)

    parser.add_argument("--use-true-kernel", action="store_true", default=False)

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

    # Set default dtype.
    torch.set_default_dtype(torch.float64)

    # Determine settings for the setup of the script.
    suffix = ""
    observe = False

    data_dir = args.data if args.mean_diff is None else f"{args.data}-{args.mean_diff}"

    # Setup script.
    if not observe:
        out.report_time = True
    wd = WorkingDirectory(
        *args.root,
        *(args.subdir or ()),
        data_dir,
        *((f"x{args.dim_x}_y{args.dim_y}",) if hasattr(args, "dim_x") else ()),
        "sgp",
        f"eps_{args.epsilon}_delta_{args.delta}",
        f"fixed_hypers_{args.use_true_kernel}",
        log=f"log{suffix}.txt",
        diff=f"diff{suffix}.txt",
        observe=observe,
    )

    # Save args.
    with open(wd.file("args.json"), "w") as f:
        json.dump(vars(args), f)

    # Determine which device to use. Try to use a GPU if one is available.
    device = "cpu"

    B.set_global_device(device)

    # General config.
    config = {
        "default": {
            "epochs": None,
            "rate": None,
            "also_ar": False,
        },
        "epsilon": args.epsilon,
        "cholesky_retry_factor": 1e6,
        "dp_epsilon_range": (args.dp_epsilon_min, args.dp_epsilon_max),
        "dp_log10_delta_range": (args.dp_log10_delta_min, args.dp_log10_delta_max),
        "min_log10_scale": args.min_log10_scale,
        "max_log10_scale": args.max_log10_scale,
        "mean_diff": args.mean_diff,
    }

    # Setup data generators for training and for evaluation.
    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=1,
        num_tasks_cv=1,
        num_tasks_eval=16,
        batch_size_eval=1,
        device=device,
    )

    if args.meta_train_hypers:
        # Run Bayesian optimisation on datasets drawn from gen_train to find hypers.
        train_x, train_y = [], []
        for _ in range(args.meta_train_size):
            batch = gen_train.generate_batch()

            xc, yc = batch["contexts"][0]
            x = xc[0, 0, :, None]
            y = xc[0, 0, :]
            train_x.append(x)
            train_y.append(y)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            partial(
                objective,
                args.epsilon,
                args.delta,
                train_x,
                train_y,
                args.use_true_kernel,
            ),
            n_trials=args.n_trials,
        )
        best_trial = study.best_trial

    for gen_name, gen in gens_eval():
        if gen_name == "interpolation in training range":
            continue

        with out.Section(gen_name.capitalize()):
            for i, batch in enumerate(gen.epoch()):
                xc, yc = batch["contexts"][0]
                xc = xc[0, 0, :, None]
                yc = yc[0, 0, :]

                xt, yt = batch["xt"], batch["yt"]
                xt = xt[0, 0, :, None]
                yt = yt[0, 0, :]

                if not args.meta_train_hypers:
                    study = optuna.create_study(direction="maximize")
                    study.optimize(
                        partial(
                            objective,
                            args.epsilon,
                            args.delta,
                            [xc],
                            [yc],
                            args.use_true_kernel,
                        ),
                        n_trials=args.n_trials,
                    )
                    best_trial = study.best_trial

                num_inducing = best_trial.params["num_inducing"]
                epochs = best_trial.params["epochs"]
                batch_size = best_trial.params["batch_size"]
                lr = best_trial.params["lr"]
                max_grad_norm = best_trial.params["max_grad_norm"]

                # Cosntruct model and optimiser using hyperparameters found via BayesOpt.
                kernel = dpsgp.kernels.RBFKernel()
                kernel.lengthscale = 0.25
                # TODO: surely we need to do BayesOpt to find optimum min/max initialisation values.
                init_z = torch.linspace(xc.min(), xc.max(), num_inducing).unsqueeze(-1)
                likelihood = dpsgp.likelihoods.GaussianLikelihood(noise=0.05)
                model = dpsgp.sgp.SparseGP(kernel, likelihood, init_z)

                train_dataset = TensorDataset(xc, yc)
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )

                privacy_engine = PrivacyEngine(accountant="rdp")
                optimiser = torch.optim.Adam(model.parameters(), lr=lr)

                (
                    model,
                    optimiser,
                    train_loader,
                ) = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimiser,
                    data_loader=train_loader,
                    epochs=epochs,
                    target_epsilon=args.epsilon,
                    target_delta=args.delta,
                    max_grad_norm=max_grad_norm,
                    grad_sample_mode="functorch",
                )

                elbo = train_model(model, optimiser, epochs, train_loader)
                visualise_1d(
                    model,
                    gen,
                    batch,
                    wd.file(f"{gen_name}-{i + 1:03d}.pdf"),
                    config,
                    args.epsilon,
                    args.delta,
                    best_trial.value,
                    elbo,
                )
                metrics = evaluate(model, xt, yt)
                for metric_name, metric_value in metrics.items():
                    file_name = wd.file(f"{gen_name}-{metric_name}-{i + 1:03d}.txt")
                    with open(file_name, "w") as f:
                        f.write(str(metric_value))
                        f.close()


if __name__ == "__main__":
    main()

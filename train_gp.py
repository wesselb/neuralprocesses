import argparse

import lab as B
import matplotlib.pyplot as plt
import neuralprocesses.torch as nps
import numpy as np
import stheno
import torch
import wbml.out as out
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak


def train(gen, objective):
    """Train for an epoch."""
    for batch in gen.epoch():
        val = B.mean(
            objective(
                batch["xc"],
                batch["yc"],
                batch["xt"],
                batch["yt"],
            )
        )
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()


def eval(gen, objective):
    """Perform evaluation."""
    with torch.no_grad():
        vals, kls, kls_diag = [], [], []
        for batch in gen.epoch():
            vs = objective(
                batch["xc"],
                batch["yc"],
                batch["xt"],
                batch["yt"],
            )

            # Save numbers, normalised by the numbers of target points.
            nt = B.shape(batch["xt"], 2)
            vals.append(B.to_numpy(vs) / nt)
            if "pred_logpdf" in batch:
                kls.append(B.to_numpy(vs + batch["pred_logpdf"]) / nt)
            if "pred_logpdf_diag" in batch:
                kls_diag.append(B.to_numpy(vs + batch["pred_logpdf_diag"]) / nt)

        # Report numbers.
        out.kv("Loglik", with_err(-B.concat(*vals)))
        if kls:
            out.kv("KL (full)", with_err(B.concat(*kls)))
        if kls_diag:
            out.kv("KL (diag)", with_err(B.concat(*kls_diag)))

        return B.mean(B.concat(*vals))


def with_err(vals):
    """Print the mean value of a list of values with error."""
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    return f"{mean:7.3f} +- {err:7.3f}"


def first_np(x):
    """Get the first batch and convert to NumPy."""
    if B.rank(x) == 2:
        return B.to_numpy(x[0, :])
    elif B.rank(x) == 3:
        return B.to_numpy(x[0, 0, :])
    elif B.rank(x) == 4:
        return B.transpose(B.to_numpy(x[0, :, 0, :]))
    else:
        raise ValueError(f"Rank must be two, three, or four.")


def plot_first_of_batch(gen, run_model):
    """Plot the prediction for the first element of a batch."""
    batch = gen.generate_batch()

    # Define points to predict at.
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), -2, 2, 500)[None, None, :]
        x = B.tile(x, B.shape(batch["xt"], 0), 1, 1)

    # Run model.
    with torch.no_grad():
        pred = run_model(batch["xc"], batch["yc"], x)
        pred_noiseless = run_model(
            batch["xc"],
            batch["yc"],
            x,
            dtype_lik=torch.float64,
            noiseless=True,
        )

    plt.figure(figsize=(6, 4))

    # Plot context and target.
    plt.scatter(
        first_np(batch["xc"]),
        first_np(batch["yc"]),
        label="Context",
        style="train",
        s=20,
    )
    plt.scatter(
        first_np(batch["xt"]),
        first_np(batch["yt"]),
        label="Target",
        style="test",
        s=20,
    )

    # Plot prediction.
    err = 1.96 * B.sqrt(pred.var)
    plt.plot(
        first_np(x),
        first_np(pred.mean),
        label="Prediction",
        style="pred",
    )
    plt.fill_between(
        first_np(x),
        first_np(pred.mean - err),
        first_np(pred.mean + err),
        style="pred",
    )
    plt.plot(
        first_np(x),
        first_np(pred_noiseless.sample(5)),
        style="pred",
        ls="-",
        lw=0.5,
    )

    # Plot prediction by ground truth.
    if hasattr(gen, "kernel"):
        f = stheno.GP(gen.kernel)
        # Make sure that everything is of `float64`s and on the GPU.
        noise = B.to_active_device(B.cast(torch.float64, gen.noise))
        xc = B.transpose(B.cast(torch.float64, batch["xc"]))
        yc = B.transpose(B.cast(torch.float64, batch["yc"]))
        x = B.transpose(B.cast(torch.float64, x))
        # Compute posterior GP.
        f_post = f | (f(xc, noise), yc)
        mean, lower, upper = f_post(x).marginal_credible_bounds()
        plt.plot(
            first_np(B.transpose(x)),
            first_np(mean),
            label="Truth",
            style="pred2",
        )
        plt.plot(
            first_np(B.transpose(x)),
            first_np(lower),
            style="pred2",
        )
        plt.plot(
            first_np(B.transpose(x)),
            first_np(upper),
            style="pred2",
        )

    plt.xlim(B.min(x), B.max(x))
    tweak()
    plt.savefig(wd.file(f"epoch-{i:03d}.pdf"))
    plt.close()


# Setup arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--dim_x", type=int, default=1)
parser.add_argument("--dim_y", type=int, default=1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--rate", type=float, default=1e-4)
parser.add_argument("--margin", type=float, default=2)
parser.add_argument("--arch", type=str)
parser.add_argument("--receptive_field", type=float, default=2)
parser.add_argument("--learnable_channel", action="store_true")
parser.add_argument("--subdir", type=str, nargs="*")
parser.add_argument(
    "--model",
    choices=["cnp", "convcnp", "convgnp-linear", "fullconvgnp"],
    required=True,
)
parser.add_argument(
    "--data",
    choices=["eq", "matern", "weakly-periodic", "sawtooth", "mixture"],
    default="eq",
)
args = parser.parse_args()

# Ensure that the `arch` is specified when it is required.
models_which_require_arch = {"convcnp", "convgnp-linear", "fullconvgnp"}
if args.model in models_which_require_arch and not args.arch:
    raise RuntimeError(f"Model requires a choice of architecture. Please set `--arch`.")

# Setup script.
out.report_time = True
B.epsilon = 1e-8
if args.learnable_channel:
    suffix = "_lc"
else:
    suffix = ""
wd = WorkingDirectory(
    "_experiments",
    *(args.subdir or ()),
    f"{args.model}",
    *((args.arch,) if args.arch else ()),
    f"x{args.dim_x}_y{args.dim_y}{suffix}",
)

# Use a GPU if one is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
B.set_global_device(device)

# Setup data.
gen = nps.construct_predefined_gens(
    torch.float32,
    seed=10,
    batch_size=args.batch_size,
    num_tasks=2**14,
    dim_x=args.dim_x,
    dim_y=args.dim_y,
    pred_logpdf=False,
    pred_logpdf_diag=False,
    device=device,
)[args.data]
gen_eval = nps.construct_predefined_gens(
    torch.float32,
    seed=20,  # Use a different seed!
    batch_size=args.batch_size,
    num_tasks=2**12,
    dim_x=args.dim_x,
    dim_y=args.dim_y,
    pred_logpdf=True,
    pred_logpdf_diag=True,
    device=device,
)[args.data]

# Setup architecture.
unet_channels = (64,) * 6
dws_channels = 128
dws_receptive_field = args.receptive_field
if args.dim_x == 1:
    points_per_unit = 64
elif args.dim_x == 2:
    # Reduce the PPU to reduce memory consumption.
    points_per_unit = 64 / 2
else:
    raise RuntimeError("Could not determine kernel for input dimensionality.")

# Construct the model.
if args.model == "convcnp":
    model = nps.construct_convgnp(
        points_per_unit=points_per_unit,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        likelihood="het",
        conv_arch=args.arch,
        unet_channels=unet_channels,
        dws_channels=dws_channels,
        dws_receptive_field=dws_receptive_field,
        margin=args.margin,
    )
    run_model = model
elif args.model == "cnp":
    model = nps.construct_gnp(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        likelihood="het",
    )
    run_model = model
elif args.model == "convgnp-linear":
    if args.learnable_channel:
        model = nps.construct_convgnp(
            points_per_unit=points_per_unit,
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            dim_yc=(args.dim_y, 128),
            likelihood="lowrank",
            conv_arch=args.arch,
            unet_channels=unet_channels,
            dws_channels=dws_channels,
            dws_receptive_field=dws_receptive_field,
            num_basis_functions=512,
            margin=args.margin,
        )

        with B.on_device(device):
            x_lc = B.linspace(torch.float32, -2, 2, 256 + 1)[None, None, :]
            x_lc = B.tile(x_lc, args.batch_size, 1, 1)
            y_lc = B.randn(torch.float32, args.batch_size, 128, 256 + 1)
        model.y_lc = model.nn.Parameter(y_lc)

        def run_model(xc, yc, xt):
            return model([(xc, yc), (x_lc, model.y_lc)], xt)

    else:

        model = nps.construct_convgnp(
            points_per_unit=points_per_unit,
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            likelihood="lowrank",
            conv_arch=args.arch,
            unet_channels=unet_channels,
            dws_channels=dws_channels,
            dws_receptive_field=dws_receptive_field,
            num_basis_functions=512,
            margin=args.margin,
        )
        run_model = model

elif args.model == "fullconvgnp":
    model = nps.construct_fullconvgnp(
        points_per_unit=points_per_unit,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        conv_arch=args.arch,
        unet_channels=unet_channels,
        dws_channels=dws_channels,
        dws_receptive_field=dws_receptive_field,
        margin=args.margin,
    )
    run_model = model
else:
    raise ValueError(f'Invalid model "{args.model}".')

# Ensure that the model is on the GPU and print some statistics.
model = model.to(device)
out.kv("Number of parameters", nps.num_params(model))


def objective(xc, yc, xt, yt):
    """Objective function."""
    # Use `float64`s for the logpdf computation.
    pred = run_model(xc, yc, xt, dtype_lik=torch.float64)
    return -pred.logpdf(B.cast(torch.float64, yt))


# Setup training loop.
opt = torch.optim.Adam(model.parameters(), args.rate)
best_eval_loss = np.inf

for i in range(args.epochs):
    with out.Section(f"Epoch {i + 1}"):
        # Perform an epoch.
        train(gen, objective)

        # Save current model.
        torch.save(model.state_dict(), wd.file(f"model-last.torch"))

        # The epoch is done. Now evaluate.
        val = eval(gen_eval, objective)

        # Check if the model is the new best. If so, save it.
        if val < best_eval_loss:
            out.out("New best model!")
            best_eval_loss = val
            torch.save(model.state_dict(), wd.file(f"model-best.torch"))

        # Visualise a prediction by the model.
        if args.dim_x == 1 and args.dim_y == 1:
            plot_first_of_batch(gen, run_model)

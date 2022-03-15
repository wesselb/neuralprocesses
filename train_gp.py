import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out as out
from mlkernels import EQ
from neuralprocesses.data import GPGenerator
from stheno import GP
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

out.report_time = True

parser = argparse.ArgumentParser()
parser.add_argument("--dim_x", type=int, default=1)
parser.add_argument("--dim_y", type=int, default=1)
parser.add_argument("--backend", choices=["tensorflow", "torch"], default="torch")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--rate", type=float, default=1e-4)
parser.add_argument(
    "--model",
    choices=["cnp", "convcnp", "convgnp-linear"],
    required=True,
)
args = parser.parse_args()

batch_size = args.batch_size
rate = args.rate
dim_x = args.dim_x
dim_y = args.dim_y

wd = WorkingDirectory("_experiments", f"{args.model}", f"x{dim_x}_y{dim_y}")

if dim_x == 1:
    kernel = EQ().stretch(0.25)
    unet_channels = (128,) * 6
    points_per_unit = 64
    margin = 7
elif dim_x == 2:
    kernel = EQ().stretch(0.25)
    unet_channels = (128,) * 6
    points_per_unit = 64 / 2
    margin = 0.1  # Cannot permit a big margin.
else:
    raise RuntimeError("Could not determine kernel for input dimensionality.")

if args.backend == "torch":

    import torch as backend
    import neuralprocesses.torch as nps

    if backend.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    def to_device(x):
        return x.to(device)

    def create_optimiser(model):
        return backend.optim.Adam(model.parameters(), rate)

    def step_optimiser(opt, model, f):
        vals = f()
        val = B.mean(vals)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()
        return vals

    def save_model(name):
        backend.save(model.state_dict(), wd.file(f"model-{name}.torch"))

elif args.backend == "tensorflow":

    import tensorflow as backend
    import neuralprocesses.tensorflow as nps

    if len(backend.config.list_physical_devices("GPU")) > 0:
        device = "gpu"
    else:
        device = "cpu"

    def to_device(x):
        return x

    def create_optimiser(model):
        return backend.keras.optimizers.Adam(rate)

    def step_optimiser(opt, model, f):
        with backend.GradientTape() as tape:
            vals = f()
            val = B.mean(vals)
        grads = tape.gradient(val, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        return vals

    def save_model(name):
        model.save_weights(wd.file(f"model-{name}.tensorflow"))

else:
    raise ValueError(f'Unknown backend "{args.backend}".')


B.set_global_device(device)

if args.model == "convcnp":
    model = nps.construct_convgnp(
        points_per_unit=points_per_unit,
        dim_x=dim_x,
        dim_y=dim_y,
        likelihood="het",
        unet_channels=unet_channels,
        margin=margin,
    )
    run_model = model
elif args.model == "cnp":
    model = nps.construct_gnp(
        dim_x=dim_x,
        dim_y=dim_y,
        likelihood="het",
    )
    run_model = model
elif args.model == "convgnp-linear":
    if dim_x == 1 and dim_y == 1:
        model = nps.construct_convgnp(
            points_per_unit=points_per_unit,
            dim_x=dim_x,
            dim_y=dim_y,
            dim_yc=(dim_y, 10),
            likelihood="lowrank",
            unet_channels=unet_channels,
            num_basis_functions=1024,
            margin=margin,
        )

        with B.on_device(device):
            x_lc = B.linspace(backend.float32, -2, 2, 256 + 1)[None, None, :]
            x_lc = B.tile(x_lc, batch_size, 1, 1)
            y_lc = B.randn(backend.float32, batch_size, 10, 256 + 1)
        model.y_lc = model.nn.Parameter(y_lc)

        def run_model(xc, yc, xt):
            return model([(xc, yc), (x_lc, model.y_lc)], xt)

    else:

        model = nps.construct_convgnp(
            points_per_unit=points_per_unit,
            dim_x=dim_x,
            dim_y=dim_y,
            likelihood="lowrank",
            unet_channels=unet_channels,
            num_basis_functions=1024,
            margin=margin,
        )

else:
    raise ValueError(f'Invalid model "{args.model}".')

model = to_device(model)
out.kv("Num. params", nps.num_params(model))

gen = GPGenerator(
    backend.float32,
    kernel=kernel,
    batch_size=batch_size,
    num_context_points=(3, 20),
    num_target_points=50,
    x_ranges=((-2, 2),) * dim_x,
    dim_y=dim_y,
    pred_logpdf=False,
    pred_logpdf_diag=False,
    device=device,
)
gen_eval = GPGenerator(
    backend.float32,
    kernel=kernel,
    num_tasks=2**12,
    batch_size=16,
    num_context_points=(3, 20),
    num_target_points=50,
    x_ranges=((-2, 2),) * dim_x,
    dim_y=dim_y,
    pred_logpdf=True,
    pred_logpdf_diag=True,
    device=device,
)


def objective(xc, yc, xt, yt):
    pred = run_model(xc, yc, xt)
    # Use `float64`s for the logpdf computation.
    pred = B.cast(backend.float64, pred)
    return -pred.logpdf(B.cast(backend.float64, yt))


def with_err(vals):
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    return f"{mean:7.3f} +- {err:7.3f}"


def first_np(x):
    if B.rank(x) == 3:
        return B.to_numpy(x[0, 0, :])
    elif B.rank(x) == 2:
        return B.to_numpy(x[0, :])
    else:
        raise ValueError(f"Rank must be two or three.")


epochs = 500
opt = create_optimiser(model)
best_eval_loss = np.inf

for i in range(epochs):
    with out.Section(f"Epoch {i + 1}"):
        # Perform an epoch.
        for batch in gen.epoch():
            step_optimiser(
                opt,
                model,
                lambda: objective(
                    batch["xc"],
                    batch["yc"],
                    batch["xt"],
                    batch["yt"],
                ),
            )

        # Save current model.
        save_model("last")

        # The epoch is done. Now evaluate.
        with backend.no_grad():
            vals = []
            full_vals = []
            diag_vals = []
            for batch in gen_eval.epoch():
                vs = objective(
                    batch["xc"],
                    batch["yc"],
                    batch["xt"],
                    batch["yt"],
                )
                nt = B.shape(batch["xt"], 2)
                vals.append(B.to_numpy(vs) / nt)
                full_vals.append(B.to_numpy(vs + batch["pred_logpdf"]) / nt)
                diag_vals.append(B.to_numpy(vs + batch["pred_logpdf_diag"]) / nt)
            vals = B.concat(*vals)
            full_vals = B.concat(*full_vals)
            diag_vals = B.concat(*diag_vals)
            out.kv("Loglik", with_err(-vals))
            out.kv("KL (diag)", with_err(diag_vals))
            out.kv("KL (full)", with_err(full_vals))

        # Check if the model is the new best.
        if B.mean(vals) < best_eval_loss:
            # Found new best model. Save it!
            out.out("New best model!")
            best_eval_loss = B.mean(vals)
            save_model("best")

        # Visualise a prediction by the model.
        if dim_x == 1 and dim_y == 1:
            with backend.no_grad():
                batch = gen_eval.generate_batch()
                with B.on_device(batch["xt"]):
                    x = B.linspace(B.dtype(batch["xt"]), -2, 2, 500)[None, None, :]
                    x = B.tile(x, B.shape(batch["xt"], 0), 1, 1)
                pred = run_model(batch["xc"], batch["yc"], x)

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
                    label="Context",
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
                    first_np(pred.sample(5)[:, 0]),
                    style="pred",
                    ls="-",
                    lw=0.5,
                )
                # Plot prediction by ground truth.
                f = GP(kernel)
                # Make sure that everything is of `float64`s and on the GPU.
                noise = B.to_active_device(B.cast(backend.float64, gen_eval.noise))
                xc = B.transpose(B.cast(backend.float64, batch["xc"]))
                yc = B.transpose(B.cast(backend.float64, batch["yc"]))
                x = B.transpose(B.cast(backend.float64, x))
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

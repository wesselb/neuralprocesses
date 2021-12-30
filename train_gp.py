import argparse

import lab as B
import numpy as np
from mlkernels import EQ
from neuralprocesses.data import GPGenerator
from wbml.experiment import WorkingDirectory
import wbml.out as out

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

wd = WorkingDirectory("_experiments", f"{args.model}")

if dim_x == 1:
    kernel = EQ().stretch(0.25)
    unet_channels = (128,) * 6
    points_per_unit = 64
elif dim_x == 2:
    kernel = EQ().stretch(0.25)
    unet_channels = (64,) * 6
    points_per_unit = 20
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

    def save_model_as_best():
        backend.save(model.state_dict(), wd.file("model.torch"))


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

    def save_model_as_best():
        model.save_weights(wd.file("model.tensorflow"))


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
    )
elif args.model == "cnp":
    model = nps.construct_gnp(
        dim_x=dim_x,
        dim_y=dim_y,
        likelihood="het",
    )
elif args.model == "convgnp-linear":
    model = nps.construct_convgnp(
        points_per_unit=points_per_unit,
        dim_x=dim_x,
        dim_y=dim_y,
        likelihood="lowrank",
        unet_channels=unet_channels,
        num_basis_functions=1024,
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
    num_tasks=2 ** 12,
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
    pred = model(xc, yc, xt)
    # Use `float64`s for the logpdf computation.
    pred = B.cast(backend.float64, pred)
    return -pred.logpdf(B.cast(backend.float64, yt))


def with_err(vals):
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    return f"{mean:7.3f} +- {err:7.3f}"


epochs = 10_000
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

        if B.mean(vals) < best_eval_loss:
            # Found new best model. Save it!
            out.out("New best model!")
            best_eval_loss = B.mean(vals)
            save_model_as_best()

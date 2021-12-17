import argparse

import lab as B
from mlkernels import EQ
from wbml.out import Progress

from neuralprocesses.data import GPGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--dim_x", type=int, default=1)
parser.add_argument("--dim_y", type=int, default=1)
parser.add_argument("--backend", choices=["tensorflow", "torch"], required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--harmonics", type=int, default=0)
args = parser.parse_args()

batch_size = args.batch_size
rate = 1e-3 * B.sqrt(args.batch_size / 16)
dim_x = args.dim_x
dim_y = args.dim_y
harmonics_range = (-2, 2) if args.harmonics > 0 else None
num_harmonics = args.harmonics

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

else:
    raise ValueError(f'Unknown backend "{args.backend}".')


B.set_global_device(device)

model = to_device(
    nps.construct_convgnp(
        points_per_unit=64,
        dim_x=dim_x,
        dim_y=dim_y,
        likelihood="lowrank",
        harmonics_range=harmonics_range,
        num_harmonics=num_harmonics,
        num_basis_functions=128,
    )
)

gen = GPGenerator(
    backend.float32,
    kernel=EQ().stretch(0.25 * B.sqrt(2) ** (dim_x - 1)),
    batch_size=batch_size,
    num_context_points=(3, 10),
    num_target_points=50,
    x_ranges=((-2, 2),) * dim_x,
    dim_y=dim_y,
    pred_logpdf=True,
    pred_logpdf_diag=True,
    device=device,
)
gen_eval = GPGenerator(
    backend.float32,
    kernel=EQ().stretch(0.25 * B.sqrt(2) ** (dim_x - 1)),
    num_tasks=4096,
    batch_size=batch_size,
    num_context_points=(3, 10),
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
    mean = B.mean(vals)
    err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    return f"{mean:7.3f} +- {err:7.3f}"


opt = create_optimiser(model)

with Progress(name="Epochs", total=10_000) as progress_epochs:
    for i in range(10_000):
        with Progress(name=f"Epoch {i + 1}", total=gen.num_batches) as progress_epoch:
            for batch in gen.epoch():
                vals = step_optimiser(
                    opt,
                    model,
                    lambda: objective(
                        batch["xc"],
                        batch["yc"],
                        batch["xt"],
                        batch["yt"],
                    ),
                )
                nt = B.shape(batch["xt"], 2)
                progress_epoch(
                    {
                        "KL (full)": with_err((vals + batch["pred_logpdf"]) / nt),
                        "KL (diag)": with_err((vals + batch["pred_logpdf_diag"]) / nt),
                    }
                )

        full_vals = []
        diag_vals = []
        for batch in gen_eval.epoch():
            batch_vals = objective(
                batch["xc"],
                batch["yc"],
                batch["xt"],
                batch["yt"],
            )
            nt = B.shape(batch["xt"], 2)
            full_vals.append(B.to_numpy(batch_vals + batch["pred_logpdf"]) / nt)
            diag_vals.append(B.to_numpy(batch_vals + batch["pred_logpdf_diag"]) / nt)
        full_vals = B.concat(*full_vals)
        diag_vals = B.concat(*diag_vals)
        progress_epochs(
            {
                "KL (full)": with_err(full_vals),
                "KL (diag)": with_err(diag_vals),
            }
        )

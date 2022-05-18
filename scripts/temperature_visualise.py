import argparse

import lab as B
import matplotlib.pyplot as plt
import torch
import wbml.out as out
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, tex, pdfcrop

import neuralprocesses.torch as nps
from neuralprocesses.mask import Masked
from train import main

parser = argparse.ArgumentParser()
parser.add_argument("--sample", action="store_true")
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
B.set_global_device(device)
tex()

wd = WorkingDirectory("_experiments", "temperature_visualise")

# Load experiment.
exp = main(
    data="temperature",
    model="convcnp-multires",
    root="aws_run_2022-05-17_temperature/_experiments",
    load=True,
)
model = exp["model"]
model.load_state_dict(
    torch.load(exp["wd"].file("model-last.torch"), map_location=device)["weights"]
)
gen = nps.TemperatureGenerator(
    torch.float32,
    seed=41,
    batch_size=1,
    subset="train",
    context_sample=True,
    device=device,
)
b = gen.generate_batch(nc=20)

# Ensure that the contexts are masked for compatibility with the below.
xc_fuse, yc_fuse = b["contexts"][0]
out.kv("Num context", B.shape(xc_fuse, -1))
if not isinstance(yc_fuse, Masked):
    yc_fuse = Masked(yc_fuse, B.ones(yc_fuse))
b["contexts"][0] = (xc_fuse, yc_fuse)

# Make predictions on a grid.
lons = B.linspace(torch.float32, 6, 16, 200)[None, None, :]
lats = B.linspace(torch.float32, 55, 47, 200)[None, None, :]
pred = model(b["contexts"], (lons, lats))

mean = pred.mean  # Mean

# Specify coarse grid to AR sample on.
n = 30
ar_lons = B.linspace(torch.float32, 6, 16, n)[:, None]
ar_lons = B.flatten(B.broadcast_to(ar_lons, n, n))[None, None, :]
ar_lats = B.linspace(torch.float32, 55, 47, n)[None, :]
ar_lats = B.flatten(B.broadcast_to(ar_lats, n, n))[None, None, :]
ar_xs = B.concat(ar_lons, ar_lats, axis=-2)

state = B.create_random_state(torch.float32, seed=0)
state, perm = B.randperm(state, torch.int64, n * n)
ar_xs = B.take(ar_xs, perm, axis=-1)

if args.sample:
    samples = []
    for _ in range(3):
        for i in range(B.shape(ar_xs, -1)):
            # Sample target.
            x = ar_xs[:, :, i : i + 1]
            y = model(b["contexts"], x).sample()

            # Append target to contexts.
            xc, yc = b["contexts"][0]
            xc = B.concat(xc, x, axis=-1)
            mask = B.concat(yc.mask, B.ones(y), axis=-1)
            yc = B.concat(yc.y, y, axis=-1)
            b["contexts"][0] = (xc, Masked(yc, mask))

        pred = model(b["contexts"], (lons, lats))
        # state, sample = pred.noiseless.sample(state)
        sample = pred.mean
        samples.append(sample)

        # Reset contexts.
        b["contexts"][0] = (xc_fuse, yc_fuse)
    wd.save(samples, "samples.pickle")
else:
    samples = wd.load("samples.pickle")

mask = yc_fuse.mask
yc_fuse = yc_fuse.y
yc_fuse[~mask] = B.nan

vmin = -15
vmax = 15
cmap = "bwr"

plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.title("Mean")
plt.imshow(
    mean[0].T,
    extent=(6, 16, 47, 55),
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
)
plt.scatter(
    xc_fuse[0, 0, :],
    xc_fuse[0, 1, :],
    c=yc_fuse[0, 0, :],
    vmin=vmin,
    vmax=vmax,
    edgecolor="white",
    lw=0.5,
    cmap=cmap,
)
tweak(legend=False, grid=False)

for i in range(len(samples)):
    plt.subplot(1, 4, 2 + i)
    plt.title(f"Sample {i + 1}")
    plt.imshow(
        samples[i][0, 0].T,
        extent=(6, 16, 47, 55),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.scatter(
        xc_fuse[0, 0, :],
        xc_fuse[0, 1, :],
        c=yc_fuse[0, 0, :],
        vmin=vmin,
        vmax=vmax,
        edgecolor="white",
        lw=0.5,
        cmap=cmap,
    )
    tweak(legend=False, grid=False)

plt.savefig(wd.file("temperature.pdf"))
pdfcrop(wd.file("temperature.pdf"))
plt.show()

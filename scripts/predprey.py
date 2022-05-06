import argparse

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import torch
import wbml.out
import wbml.out as out
from wbml.data.predprey import load
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak, tex, pdfcrop

import neuralprocesses.torch as nps
from train import main

# Setup script.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--ar", action="store_true")
args = parser.parse_args()

wbml.out.report_time = True

# Load experiment.
with out.Section("Loading experiment"):
    exp = main(
        # The keyword arguments here must line up with the arguments you provide on the
        # command line.
        model=args.model,
        data="predprey",
        # Point `root` to where the `_experiments` directory is located. In my case,
        # I'm `rsync`ing it to the directory `server`.
        root="server/_experiments",
        load=True,
    )
    model = exp["model"]
    model.load_state_dict(
        torch.load(exp["wd"].file("model-last.torch"), map_location="cpu")["weights"]
    )

# Setup another working directory to save output of the evaluation in.
wd = WorkingDirectory("_experiments", "eval", "predprey", args.model)
tex()

# Increase regularisation.
B.epsilon = 1e-6

# Load the data.
df = load()
x = torch.tensor(np.array(df.index), dtype=torch.float32)
x = x - x[0]
y = torch.tensor(np.array(df[["hare", "lynx"]]), dtype=torch.float32)

# Construct mask for the hares.
mask_hare = x < 0
for i in range(20, 80, 8):
    mask_hare |= (x >= i) & (x <= i + 3)
mask_hare = ~mask_hare

# Construct a mask for the lynxes.
mask_lynx = x < 0
for i in range(25, 80, 8):
    mask_lynx |= (x >= i) & (x <= i + 3)
mask_lynx = ~mask_lynx

# Share tensors into the standard formats.
x = x[None, None, :]
y = y.T[None, :, :]
xt = torch.linspace(-20, 120, 141, dtype=torch.float32)[None, None, :]

contexts = [
    (x[:, :, mask_hare], y[:, 0:1, mask_hare]),
    (x[:, :, mask_lynx], y[:, 1:2, mask_lynx]),
]

# `torch.no_grad` is necessary to prevent memory from accumulating.
with torch.no_grad():

    # Perform evaluation.
    xt_eval = nps.AggregateInput((x[:, :, ~mask_hare], 0), (x[:, :, ~mask_lynx], 1))
    yt_eval = nps.Aggregate(y[:, 0:1, ~mask_hare], y[:, 1:2, ~mask_lynx])
    out.kv(
        "Logpdf",
        nps.loglik(model, contexts, xt_eval, yt_eval, normalise=True),
    )
    if args.ar:
        out.kv(
            "Logpdf (AR)",
            nps.ar_loglik(model, contexts, xt_eval, yt_eval, normalise=True),
        )

    # Make predictions.
    predict = nps.ar_predict if args.ar else nps.predict
    mean, _, noiseless_samples, noisy_samples = predict(
        model,
        contexts,
        nps.AggregateInput((xt, 0), (xt, 1)),
        num_samples=100,
    )

# Plot the result.

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.scatter(
    x[0, 0, mask_hare],
    y[0, 0, mask_hare],
    marker="o",
    style="train",
    s=20,
    label="Hare",
)
plt.scatter(x[0, 0, ~mask_hare], y[0, 0, ~mask_hare], marker="o", style="test", s=20)
plt.plot(xt[0, 0], mean[0][0, 0, :], style="pred")
plt.plot(xt[0, 0], noiseless_samples[0][:10, 0, 0, :].T, style="pred", ls="-", lw=0.5)
plt.fill_between(
    xt[0, 0],
    B.quantile(noisy_samples[0][:, 0, 0, :], 2.5 / 100, axis=0),
    B.quantile(noisy_samples[0][:, 0, 0, :], (100 - 2.5) / 100, axis=0),
    style="pred",
)
plt.ylim(0, 300)
tweak()

plt.subplot(2, 1, 2)
plt.scatter(
    x[0, 0, mask_lynx],
    y[0, 1, mask_lynx],
    marker="o",
    style="train",
    s=20,
    label="Lynx",
)
plt.scatter(x[0, 0, ~mask_lynx], y[0, 1, ~mask_lynx], marker="o", style="test", s=20)
plt.plot(xt[0, 0], mean[1][0, 0, :], style="pred")
plt.plot(xt[0, 0], noiseless_samples[1][:10, 0, 0, :].T, style="pred", ls="-", lw=0.5)
plt.fill_between(
    xt[0, 0],
    B.quantile(noisy_samples[1][:, 0, 0, :], 2.5 / 100, axis=0),
    B.quantile(noisy_samples[1][:, 0, 0, :], (100 - 2.5) / 100, axis=0),
    style="pred",
)
plt.ylim(0, 300)
tweak()

plt.savefig(wd.file("predprey.pdf"))
pdfcrop(wd.file("predprey.pdf"))
plt.show()

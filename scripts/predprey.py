import lab as B
import matplotlib.pyplot as plt
import numpy as np
import torch
import wbml.out as out
from wbml.data.predprey import load
from wbml.plot import tweak, tex, pdfcrop
from wbml.experiment import WorkingDirectory

import neuralprocesses.torch as nps
from train import main

# Load experiment.
with out.Section("Loading experiment"):
    exp = main(
        # The keyword arguments here must line up with the arguments you provide on the
        # command line.
        model="fullconvgnp",
        data="predprey",
        # Point `root` to where the `_experiments` directory is located. In my case,
        # I'm `rsync`ing it to the directory `server`.
        root="server/_experiments",
        load=True,
    )
    model = exp["model"]
    model.load_state_dict(
        torch.load(exp["wd"].file("model-last.torch"), map_location="cpu")
    )

# Setup another working directory to save plot in.
wd = WorkingDirectory("_experiments", "eval", "predprey")
tex()

# Increase regularisation.
B.epsilon = 1e-5

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
xt = torch.linspace(-20, 120, 1000, dtype=torch.float32)[None, None, :]

contexts = [
    (x[:, :, mask_hare], y[:, 0:1, mask_hare]),
    (x[:, :, mask_lynx], y[:, 1:2, mask_lynx]),
]

# Perform evaluation.
xt_eval = nps.AggregateInput(
    (x[:, :, ~mask_hare], 0),
    (x[:, :, ~mask_lynx], 1),
)
yt_eval = nps.Aggregate(
    B.cast(torch.float64, y[:, 0:1, ~mask_hare]),
    B.cast(torch.float64, y[:, 1:2, ~mask_lynx]),
)
logpdf = model(
    contexts, xt_eval, dtype_enc_sample=torch.float32, dtype_lik=torch.float64
).logpdf(yt_eval) / nps.num_data(xt_eval, yt_eval)
out.kv("Eval logpdf", B.mean(logpdf))

# Make predictions.
pred = model(contexts, xt, dtype_enc_sample=torch.float32, dtype_lik=torch.float64)
samples = pred.sample(10_000)
mean = B.mean(samples, axis=0)
lower = B.quantile(samples, 2.5 / 100, axis=0)
upper = B.quantile(samples, (100 - 2.5) / 100, axis=0)

# Make some noiseless samples.
pred_noiseless = model(
    contexts,
    xt,
    dtype_enc_sample=torch.float32,
    dtype_lik=torch.float64,
    noiseless=True,
)
samples = pred_noiseless.sample(5)

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
plt.plot(xt[0, 0], mean[0, 0, :], style="pred")
plt.plot(xt[0, 0], samples[:5, 0, 0, :].T, style="pred", ls="-", lw=0.5)
err = 1.96 * B.sqrt(pred.var[0, 0, :])
plt.fill_between(xt[0, 0], lower[0, 0, :], upper[0, 0, :], style="pred")
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
plt.plot(xt[0, 0], mean[0, 1, :], style="pred")
plt.plot(xt[0, 0], samples[:5, 0, 1, :].T, style="pred", ls="-", lw=0.5)
err = 1.96 * B.sqrt(pred.var[0, 1, :])
plt.fill_between(xt[0, 0], lower[0, 1, :], upper[0, 1, :], style="pred")
plt.ylim(0, 300)
tweak()

plt.savefig(wd.file("predprey.pdf"))
pdfcrop(wd.file("predprey.pdf"))
plt.show()

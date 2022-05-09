import argparse

import lab as B
import matplotlib.pyplot as plt
import neuralprocesses.torch as nps
import torch
import wbml.out as out
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

# Parse the arguments, which should include the path to the weights.
parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
args = parser.parse_args()

# Setup a directory to store the results in.
out.report_time = True
wd = WorkingDirectory("_experiments", "sawtooth_sample_ar")

# Construct the model and load the weights.
model = nps.construct_convgnp(
    dim_x=1,
    dim_y=1,
    unet_channels=(64,) * 6,
    points_per_unit=64,
    likelihood="het",
)
model.load_state_dict(torch.load(args.weights, map_location="cpu")["weights"])

# Construct the data generator the model was trained on.
gen = nps.SawtoothGenerator(
    torch.float32,
    seed=2,
    batch_size=1,  # Only need one sample.
    # Use only two context points to introduce ambiguity.
    num_context=nps.UniformDiscrete(2, 2),
    # Be sure to use the same distribution of frequencies we used during training.
    dist_freq=nps.UniformContinuous(2, 4),
)
batch = gen.generate_batch()  # Sample a batch of data.

# Predict at the following points.
x = B.linspace(torch.float32, -2, 2, 400)[None, None, :]
pred = model(batch["contexts"], x)

plt.figure(figsize=(12, 6))

for i, order in enumerate(["random", "left-to-right"]):
    # We can predict autoregressively by using `nps.ar_predict`.
    mean, var, noiseless_samples, noisy_samples = nps.ar_predict(
        model,
        batch["contexts"],
        # The inputs to predict at need to be wrapped in a `nps.AggregateInput`. For
        # this particular problem, this extra functionality is not needed, but it is
        # needed in the case of multiple outputs. Below, `(x, 0)` means to predict
        # output 0 at inputs `x`. The return values will also be wrapped in
        # `nps.Aggregate`s, which can be accessed by indexing with the output index,
        # 0 in this case.
        nps.AggregateInput((x, 0)),
        num_samples=6,
        order=order,
    )

    for j, (title_suffix, samples) in enumerate(
        [("noiseless", noiseless_samples), ("noisy", noisy_samples)]
    ):
        plt.subplot(2, 2, 2 * i + 1 + j)
        plt.title(order.capitalize() + f" ({title_suffix})")
        # Plot the context points.
        plt.scatter(
            nps.batch_xc(batch, 0)[0, 0],
            nps.batch_yc(batch, 0)[0],
            style="train",
        )
        # Plot the mean and variance of the non-AR predction.
        plt.plot(x[0, 0], pred.mean[0, 0], style="pred")
        err = 1.96 * B.sqrt(pred.var[0, 0])
        plt.fill_between(
            x[0, 0],
            pred.mean[0, 0] - err,
            pred.mean[0, 0] + err,
            style="pred",
        )
        # Plot the samples.
        plt.plot(x[0, 0], samples[0][:, 0, 0, :].T, style="pred", lw=0.5, ls="-")
        plt.ylim(-0.2, 1.2)
        plt.xlim(-2, 2)
        tweak()

plt.savefig(wd.file("samples.pdf"))
plt.show()

import sys

import experiment
import lab as B
import torch
import wbml.out as out
from neuralprocesses import AugmentedInput
from train import main

# Load experiment.
sys.argv += ["--load"]
exp = main()

# Setup model.
model = exp["model"]
model.load_state_dict(torch.load(exp["wd"].file("model-last.torch"))["weights"])


def strip_augmentation(x):
    """Strip possible augmentation from the inputs."""
    if isinstance(x, AugmentedInput):
        return x.x
    return x


xt_all = strip_augmentation(exp["gens_eval"]()[0][1].generate_batch()["xt"])[0]


def reindex(mae, xt):
    """Let the MAEs to correctly line up for randomly sampled batched."""
    nan_row = mae[:, :, :1] * B.nan
    xt = strip_augmentation(xt)[0]

    # Precomputing the distances like this allows us to get away with a simple
    # `for`-loop below. No need to optimise that further.
    dists = B.to_numpy(B.pw_dists(B.t(xt_all), B.t(xt)).cpu())

    rows = []
    for i in range(B.shape(xt_all, -1)):
        match = False
        for j in range(B.shape(xt, -1)):
            if dists[i, j] < 1e-6:
                rows.append(mae[:, :, j : j + 1])
                match = True
                break
        if not match:
            rows.append(nan_row)
    return B.concat(*rows, axis=-1)


for name, gen in exp["gens_eval"]():
    with out.Section(name):
        state = B.create_random_state(torch.float32, seed=0)
        maes = []

        with torch.no_grad():
            for batch in gen.epoch():
                state, pred = model(state, batch["contexts"], batch["xt"])
                mae = B.abs(pred.mean - batch["yt"])
                maes.append(reindex(mae, batch["xt"]))
        maes = B.concat(*maes)

        # Compute the average MAE per station, and then take the median over
        # stations. This lines up with the VALUE protocol.
        maes = B.nanmean(maes, axis=(0, 1))

        out.kv("Station-wise MAEs", maes)
        out.kv("MAE", experiment.with_err(maes[~B.isnan(maes)]))
        out.kv("MAE (median)", experiment.with_err(*experiment.median_and_err(maes)))

import sys

import lab as B
import torch
import wbml.out as out

from train import main

# Load experiment.
sys.argv += ["--load"]
exp = main()

# Setup model.
model = exp["model"]
model.load_state_dict(torch.load(exp["wd"].file("model-best.torch"))["weights"])

# Setup generator.
gens = exp["gens_eval"]()
_, gen = gens[0]  # The first one corresponds to downscaling.

state = B.create_random_state(torch.float32, seed=0)
maes = []
with torch.no_grad():
    for batch in gen.epoch():
        state, pred = model(state, batch["contexts"], batch["xt"])
        maes.append(B.abs(pred.mean - batch["yt"]))
maes = B.concat(*maes)

# Compute the average MAE per station, and then take the median over
# stations. This lines up with the VALUE protocol.
mae = torch.median(B.nanmean(maes, axis=(0, 1)))

out.kv("MAE", mae)

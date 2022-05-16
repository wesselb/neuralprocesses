import sys

import lab as B
import torch
import wbml.out as out

from experiment.util import with_err
from train import main

sys.argv += "--load"

# Load experiment.
exp = main()
model = exp["model"]
model.load_state_dict(torch.load(exp["wd"].file("model-last.torch"))["weights"])

maes = []
state = B.create_random_state(torch.float32, seed=0)

for batch in exp["gen_cv"]:
    state, pred = model(state, batch["contexts"], batch["xt"])
    mask = ~B.isnan(batch["yt"])
    maes.append(B.abs(pred.mean[mask] - batch["yt"][mask]))

out.kv("MAE", with_err(B.concat(*maes), and_upper=True))

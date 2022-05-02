import lab as B
import neuralprocesses.torch as nps
import torch
import matplotlib.pyplot as plt
from wbml.plot import tweak


gen = nps.PredPreyGenerator(
    torch.float32,
    seed=0,
    num_target=nps.UniformDiscrete(200, 200),
)
batch = gen.generate_batch()

plt.figure(figsize=(12, 8))

for i in range(16):
    # Plot the preys (output 0).
    plt.subplot(4, 4, i + 1)
    xt = nps.batch_xt(batch, 0)[i, 0]
    yt = nps.batch_yt(batch, 0)[i]
    inds = B.argsort(xt)  # Sort before plotting to prevent a mess.
    plt.plot(xt[inds], yt[inds])

    #  Plot the predators (output 1).
    xt = nps.batch_xt(batch, 1)[i, 0]
    yt = nps.batch_yt(batch, 1)[i]
    inds = B.argsort(xt)
    plt.plot(xt[inds], yt[inds])

    tweak()

plt.show()

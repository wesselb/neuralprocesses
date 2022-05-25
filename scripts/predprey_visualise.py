import argparse

import matplotlib.pyplot as plt
import torch
from wbml.plot import tweak

import neuralprocesses.torch as nps

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str)
args = parser.parse_args()

gen = nps.PredPreyGenerator(torch.float32, seed=0, mode=args.mode)
batch = gen.generate_batch()

plt.figure(figsize=(12, 8))

for i in range(16):
    # Plot the preys (output 0).
    plt.subplot(4, 4, i + 1)
    xc = nps.batch_xc(batch, 0)[i, 0]
    yc = nps.batch_yc(batch, 0)[i]
    xt = nps.batch_xt(batch, 0)[i, 0]
    yt = nps.batch_yt(batch, 0)[i]
    plt.scatter(xc, yc, c="tab:red", marker="x", s=5)
    plt.scatter(xt, yt, c="tab:orange", marker="x", s=5)

    #  Plot the predators (output 1).
    xc = nps.batch_xc(batch, 1)[i, 0]
    yc = nps.batch_yc(batch, 1)[i]
    xt = nps.batch_xt(batch, 1)[i, 0]
    yt = nps.batch_yt(batch, 1)[i]
    plt.scatter(xc, yc, c="tab:blue", marker="o", s=5)
    plt.scatter(xt, yt, c="tab:cyan", marker="o", s=5)

    plt.xlim(0, 100)
    tweak()

plt.show()

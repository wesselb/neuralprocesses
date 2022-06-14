import argparse
import logging
import logging

import matplotlib.pyplot as plt
import torch
import lab as B
import numpy as np

from neuralprocesses.data.phone import PhoneGenerator
from .util import nps

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def test_generate_batch(nps):
    gen = PhoneGenerator(
        torch.float32,
        # seed=B.randint(lower=0, upper=1000).item(),
        seed=2,
        batch_size=16,  # Only need one sample.
        # ↓ guessing from distribution of utterance lengths
        num_target=nps.UniformDiscrete(250, 500),
        num_data=nps.UniformDiscrete(500, 1500),
        # num_target=nps.UniformDiscrete(1000, 1000), # to see full plot
        # num_data=nps.UniformDiscrete(2000, 2000),  # to see full plot
        data_path="../../data/phn0.npy",
        # TODO: data_path should be included in a more stable way
    )
    batch = gen.generate_batch()  # Sample a batch of data.
    contexts = batch["contexts"]
    # i = np.random.randint(0, batch["xt"][0][0].shape[0])
    i = 12
    x = batch["xt"].elements[0][0][i].numpy().reshape(-1)
    y = batch["yt"].elements[0][i].numpy().reshape(-1)
    # plt.scatter(x, y)
    order = np.argsort(x)  # might need to keep track of dim more carefully
    sort_x = x[order]
    sort_y = y[order]
    # plt.scatter(sort_x, sort_y)
    # plt.plot(sort_x, sort_y)
    # plt.xlabel("Frame")
    # plt.ylabel("Amplitude")
    # plt.show()
    max_y = sort_y.max()
    assert np.isclose(max_y, 0.023, atol=0.001)

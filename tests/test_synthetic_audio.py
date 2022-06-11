import logging

import matplotlib.pyplot as plt
import torch
import lab as B

from neuralprocesses.data.synthetic_audio import SoundlikeGenerator
from .util import nps

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def test_generate_batch(nps):
    gen = SoundlikeGenerator(
        torch.float32,
        # seed=B.randint(lower=0, upper=1000).item(),
        seed=2,
        batch_size=1,  # Only need one sample.
        # Use only two context points to introduce ambiguity.
        num_context=nps.UniformDiscrete(1, 1),
        # Be sure to use the same distribution of frequencies we used during training.
        # dist_freq=nps.UniformContinuous(2, 4),
        noise=0.001,
        dist_w1=nps.UniformContinuous(51, 51),
        dist_w2=nps.UniformContinuous(65, 65),
        num_target=nps.UniformDiscrete(2000, 2000),
    )
    batch = gen.generate_batch()  # Sample a batch of data.
    contexts = batch["contexts"]
    # plt.scatter(batch["xt"].numpy().reshape(-1), batch["yt"].numpy().reshape(-1))
    order = B.argsort(batch['xt'])  # might need to keep track of dim more carefully
    sort_x = batch['xt'][..., order]
    sort_y = batch['yt'][..., order]
    # plt.plot(sort_x.numpy().reshape(-1), sort_y.numpy().reshape(-1))
    # plt.show()
    a = torch.Tensor([2.0])
    b = torch.max(sort_y)
    print(a, b)
    assert torch.isclose(a, b, atol=0.1)

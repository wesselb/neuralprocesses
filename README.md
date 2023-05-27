# [Neural Processes](http://github.com/wesselb/neuralprocesses)

[![CI](https://github.com/wesselb/neuralprocesses/workflows/CI/badge.svg)](https://github.com/wesselb/neuralprocesses/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/neuralprocesses/badge.svg?branch=main)](https://coveralls.io/github/wesselb/neuralprocesses?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/neuralprocesses)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A framework for composing Neural Processes in Python.

## Installation

```
pip install neuralprocesses
```

If something is not working or unclear, please feel free to open an issue.

## Documentation

See [here](https://wesselb.github.io/neuralprocesses).

## TL;DR! Just Get me Started!

Here you go:

```python
import torch

import neuralprocesses.torch as nps

# Construct a ConvCNP.
convcnp = nps.construct_convgnp(dim_x=1, dim_y=2, likelihood="het")

# Construct optimiser.
opt = torch.optim.Adam(convcnp.parameters(), 1e-3)

# Training: optimise the model for 32 batches.
for _ in range(32):
    # Sample a batch of new context and target sets. Replace this with your data. The
    # shapes are `(batch_size, dimensionality, num_data)`.
    xc = torch.randn(16, 1, 10)  # Context inputs
    yc = torch.randn(16, 2, 10)  # Context outputs
    xt = torch.randn(16, 1, 15)  # Target inputs
    yt = torch.randn(16, 2, 15)  # Target output

    # Compute the loss and update the model parameters.
    loss = -torch.mean(nps.loglik(convcnp, xc, yc, xt, yt, normalise=True))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

# Testing: make some predictions.
mean, var, noiseless_samples, noisy_samples = nps.predict(
    convcnp,
    torch.randn(16, 1, 10),  # Context inputs
    torch.randn(16, 2, 10),  # Context outputs
    torch.randn(16, 1, 15),  # Target inputs
)
```

# [Neural Processes](http://github.com/wesselb/neuralprocesses)

[![CI](https://github.com/wesselb/neuralprocesses/workflows/CI/badge.svg)](https://github.com/wesselb/neuralprocesses/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/neuralprocesses/badge.svg?branch=main)](https://coveralls.io/github/wesselb/neuralprocesses?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/neuralprocesses)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*The package and manual are still under construction.*
*If something is not working or unclear, please feel free to open an issue.*

A framework for composing Neural Processes in Python.
See also [NeuralProcesses.jl](https://github.com/wesselb/NeuralProcesses.jl).

Contents:

- [Installation](#installation)
- [Examples of Predefined Models](#examples-of-predefined-models)
  - [PyTorch](#pytorch)
    - [GNP](#gnp)
    - [ConvGNP](#convgnp)
  - [TensorFlow](#tensorflow)
    - [GNP](#gnp-1)
    - [ConvGNP](#convgnp-1)
    - [ConvGNP With Auxiliary Variables](#convgnp-with-auxiliary-variables)
- [Masking](#masking)
    - [Masking Particular Inputs](#masking-particular-inputs)
    - [Using Masks to Batch Contexts of Different Sizes](#using-masks-to-batch-context-sets-of-different-sizes)
- [Build Your Own Model](#build-your-own-model)
  - [ConvGNP](#convgnp-2)
  - [ConvGNP With Auxiliary Variables](#convgnp-with-auxiliary-variables-1)

## Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```
pip install neuralprocesses
```

## Manual

Inputs and outputs are always tensors with shape `(b, d, n)` where `b` is the
batch size, `d` is the dimensionality of the inputs/outputs, and `n` is the number of data
points.

## Examples of Predefined Models

### PyTorch

#### GNP

```python
import lab as B
import torch

import neuralprocesses.torch as nps

cnp = nps.construct_gnp(dim_x=2, dim_y=3, likelihood="lowrank")
dist = cnp(
    B.randn(torch.float32, 16, 2, 10),  # Context inputs
    B.randn(torch.float32, 16, 3, 10),  # Context outputs
    B.randn(torch.float32, 16, 2, 15),  # Target inputs
)
mean, var = dist.mean, dist.var  # Prediction for target outputs

print(dist.logpdf(B.randn(torch.float32, 16, 3, 15)))
print(dist.sample())
print(dist.kl(dist))
print(dist.entropy())
```

#### ConvGNP

```python
import lab as B
import torch

import neuralprocesses.torch as nps

cnp = nps.construct_convgnp(dim_x=2, dim_y=3, likelihood="lowrank")
dist = cnp(
    B.randn(torch.float32, 16, 2, 10),  # Context inputs
    B.randn(torch.float32, 16, 3, 10),  # Context outputs
    B.randn(torch.float32, 16, 2, 15),  # Target inputs
)
mean, var = dist.mean, dist.var  # Prediction for target outputs

print(dist.logpdf(B.randn(torch.float32, 16, 3, 15)))
print(dist.sample())
print(dist.kl(dist))
print(dist.entropy())
```

### TensorFlow

#### GNP

```python
import lab as B
import tensorflow as tf

import neuralprocesses.tensorflow as nps

cnp = nps.construct_gnp(dim_x=2, dim_y=3, likelihood="lowrank")
dist = cnp(
    B.randn(tf.float32, 16, 2, 10),  # Context inputs
    B.randn(tf.float32, 16, 3, 10),  # Context outputs
    B.randn(tf.float32, 16, 2, 15),  # Target inputs
)
mean, var = dist.mean, dist.var  # Prediction for target outputs

print(dist.logpdf(B.randn(tf.float32, 16, 3, 15)))
print(dist.sample())
print(dist.kl(dist))
print(dist.entropy())
```

#### ConvGNP

```python
import lab as B
import tensorflow as tf

import neuralprocesses.tensorflow as nps

cnp = nps.construct_convgnp(dim_x=2, dim_y=3, likelihood="lowrank")
dist = cnp(
    B.randn(tf.float32, 16, 2, 10),  # Context inputs
    B.randn(tf.float32, 16, 3, 10),  # Context outputs
    B.randn(tf.float32, 16, 2, 15),  # Target inputs
)
mean, var = dist.mean, dist.var  # Prediction for target outputs

print(dist.logpdf(B.randn(tf.float32, 16, 3, 15)))
print(dist.sample())
print(dist.kl(dist))
print(dist.entropy())
```

#### ConvGNP with Auxiliary Variables

```python
import lab as B
import tensorflow as tf

import neuralprocesses.tensorflow as nps

cnp = nps.construct_convgnp(
    dim_x=2,
    dim_yc=(
        3,  # Observed data has three dimensions.
        1,  # First auxiliary variable has one dimension.
        2,  # Second auxiliary variable has two dimensions.
    ),
    # Third auxiliary variable has four dimensions and is auxiliary information specific
    # to the target inputs.
    dim_aux_t=4,
    dim_yt=3,  # Predictions have three dimensions.
    num_basis_functions=64, 
    likelihood="lowrank",
)

observed_data = (
    B.randn(tf.float32, 16, 2, 10),
    B.randn(tf.float32, 16, 3, 10),
)

# Define three auxiliary variables. The first one is specified like the observed data
# at arbitrary inputs.
aux_var1 = (
    B.randn(tf.float32, 16, 2, 12),
    B.randn(tf.float32, 16, 1, 12),  # Has one dimension.
)
# The second one is specified on a grid.
aux_var2 = (
    (B.randn(tf.float32, 16, 1, 25), B.randn(tf.float32, 16, 1, 35)),
    B.randn(tf.float32, 16, 2, 25, 35),  # Has two dimensions.
)
# The third one is specific to the target inputs. We could encode it like the first
# auxiliary variable `aux_var1`, but we illustrate how an MLP-style encoding can
# also be used. The number must match the number of target inputs!
aux_var_t = B.randn(tf.float32, 16, 4, 15)  # Has four dimensions.

dist = cnp(
    [observed_data, aux_var1, aux_var2],
    B.randn(tf.float32, 16, 2, 15),
    aux_t=aux_var_t,  # This must be given as a keyword argument.
)
mean, var = dist.mean, dist.var

print(dist.logpdf(B.randn(tf.float32, 16, 3, 15)))
print(dist.sample())
print(dist.kl(dist))
print(dist.entropy())
```

## Masking

In this section, we'll take the following ConvGNP as a running example:

```python
import lab as B
import torch

import neuralprocesses.torch as nps

cnp = nps.construct_convgnp(
    dim_x=2,
    dim_yc=(1, 1),  # Two context sets, both with one channel
    dim_yt=1, 
)

# Construct two sample context sets with one on a grid.
xc = B.randn(torch.float32, 1, 2, 20)
yc = B.randn(torch.float32, 1, 1, 20)
xc_grid = (B.randn(torch.float32, 1, 1, 10), B.randn(torch.float32, 1, 1, 15))
yc_grid = B.randn(torch.float32, 1, 1, 10, 15)

# Contruct sample target inputs
xt = B.randn(torch.float32, 1, 2, 50)
```

For example, then predictions can be made via

```python
>>> pred = cnp([(xc, yc), (xc_grid, yc_grid)], xt)
```

### Masking Particular Inputs

Suppose that due to a particular reason you didn't observe `yc_grid[5, 5]`.
In the specification above, it is not possible to just omit that one element.
The proposed solution is to use a _mask_.
A mask `mask` is a tensor of the same size as the context outputs (`yc_grid` in this case)
but with _only one channel_ consisting of ones and zeros.
If `mask[i, 0, j, k] = 1`, then that means that `yc_grid[i, :, j, k]` is observed.
On the other hand, if `mask[i, 0, j, k] = 0`, then that means that `yc_grid[i, :, j, k]`
is _not_ observed.
`yc_grid[i, :, j, k]` will still have values, _which must be not NaNs_, but those values
will be ignored.
To mask context outputs, use `nps.Masked(yc_grid, mask)`.

Definition:

```python
masked_yc = Masked(yc, mask)
```

Example:

```python
>>> mask = B.ones(torch.float32, 1, 1, *B.shape(yc_grid, 2, 3))

>>> mask[:, :, 5, 5] = 0

>>> pred = cnp([(xc, yc), (xc_grid, nps.Masked(yc_grid, mask))], xt)
```

Masking is also possible for non-gridded contexts.

Example:

```python
>>> mask = B.ones(torch.float32, 1, 1, B.shape(yc, 2))

>>> mask[:, :, 2:7] = 0   # Elements 3 to 7 are missing.

>>> pred = cnp([(xc, nps.Masked(yc, mask)), (xc_grid, yc_grid)], xt)
```

### Using Masks to Batch Context Sets of Different Sizes

Suppose that we also had another context set, of a different size:

```python
# Construct another two sample context sets with one on a grid.
xc2 = B.randn(torch.float32, 1, 2, 30)
yc2 = B.randn(torch.float32, 1, 1, 30)
xc2_grid = (B.randn(torch.float32, 1, 1, 5), B.randn(torch.float32, 1, 1, 20))
yc2_grid = B.randn(torch.float32, 1, 1, 5, 20)
```

Rather than running the model once for `[(xc, yc), (xc_grid, yc_grid)]` and once for 
`[(xc2, yc2), (xc2_grid, yc2_grid)]`, we would like to concatenate the
two context sets along the batch dimension and run the model only once.
This, however, doesn't work, because the twe context sets have different sizes.

The proposed solution is to pad the context sets with zeros to align them, concatenate
the padded contexts, and use a mask to reject the padded zeros.
The function `nps.merge_contexts` can be used to do this automatically.

Definition:

```python
xc_merged, yc_merged = nps.merge_contexts((xc1, yc1), (xc2, yc2), ...)
```

Example:

```python
xc_merged, yc_merged = nps.merge_contexts((xc, yc), (xc2, yc2))
xc_grid_merged, yc_grid_merged = nps.merge_contexts(
    (xc_grid, yc_grid), (xc2_grid, yc2_grid)
)
```

```python
>>> pred = cnp(
    [(xc_merged, yc_merged), (xc_grid_merged, yc_grid_merged)],
    B.concat(xt, xt, axis=0)
)
```

## Build Your Own Model

### ConvGNP

```python
import lab as B
import tensorflow as tf

import neuralprocesses.tensorflow as nps

dim_x = 1
dim_y = 1

# CNN architecture:
unet = nps.UNet(
    dim=dim_x,
    in_channels=2 * dim_y,
    out_channels=(2 + 512) * dim_y,
    channels=(8, 16, 16, 32, 32, 64),
)

# Discretisation of the functional embedding:
disc = nps.Discretisation(
    points_per_unit=64,
    multiple=2**unet.num_halving_layers,
    margin=0.1,
    dim=dim_x,
)

# Create the encoder and decoder and construct the model.
encoder = nps.FunctionalCoder(
    disc,
    nps.Chain(
        nps.PrependDensityChannel(),
        nps.SetConv(scale=2 / disc.points_per_unit),
        nps.DivideByFirstChannel(),
        nps.DeterministicLikelihood(),
    ),
)
decoder = nps.Chain(
    unet,
    nps.SetConv(scale=2 / disc.points_per_unit),
    nps.LowRankGaussianLikelihood(512),
)
convgnp = nps.Model(encoder, decoder)

# Run the model on some random data.
dist = convgnp(
    B.randn(tf.float32, 16, 1, 10),
    B.randn(tf.float32, 16, 1, 10),
    B.randn(tf.float32, 16, 1, 15),
)
```

### ConvGNP with Auxiliary Variables

```python
import lab as B
import tensorflow as tf

import neuralprocesses.tensorflow as nps

dim_x = 2
# We will use two target sets with output dimensionalities `dim_y` and `dim_y2`.
dim_y = 1
dim_y2 = 10
# We will also use auxiliary target information of dimensionality `dim_aux_t`.
dim_aux_t = 7

# CNN architecture:
unet = nps.UNet(
    dim=dim_x,
    # The outputs are `dim_y`-dimensional, and we will use another context set
    # consisting of `dim_y2` variables. Both of these context sets will also have a
    # density channel.
    in_channels=dim_y + 1 + dim_y2 + 1,
    out_channels=8,
    channels=(8, 16, 16, 32, 32, 64),
)

# Discretisation of the functional embedding:
disc = nps.Discretisation(
    points_per_unit=32,
    multiple=2**unet.num_halving_layers,
    margin=0.1,
    dim=dim_x,
)

# Create the encoder and decoder and construct the model.
encoder = nps.FunctionalCoder(
    disc,
    nps.Chain(
        nps.PrependDensityChannel(),
        # Use a separate set conv for every context set. Here we initialise the length
        # scales of these set convs both to `2 / disc.points_per_unit`.
        nps.Parallel(
            nps.SetConv(scale=2 / disc.points_per_unit),
            nps.SetConv(scale=2 / disc.points_per_unit),
        ),
        nps.DivideByFirstChannel(),
        # Concatenate the encodings of the context sets.
        nps.Materialise(),
        nps.DeterministicLikelihood(),
    ),
)
decoder = nps.Chain(
    unet,
    nps.SetConv(scale=2 / disc.points_per_unit),
    # `nps.Augment` will concatenate any auxiliary information to the current encoding
    # before proceedings.
    nps.Augment(
        nps.Chain(
            nps.MLP(
                # Input dimensionality is equal to the number of channels coming out of
                # `unet` plus the dimensionality of the auxiliary target information.
                in_dim=8 + dim_aux_t,
                layers=(128,) * 3,
                out_dim=(2 + 512) * dim_y,
            ),
            nps.LowRankGaussianLikelihood(512),
        )
    )
)
convgnp = nps.Model(encoder, decoder)

# Run the model on some random data.
dist = convgnp(
    [
        (
            B.randn(tf.float32, 16, dim_x, 10),
            B.randn(tf.float32, 16, dim_y, 10),
        ),
        (
            # The second context set is given on a grid.
            (B.randn(tf.float32, 16, 1, 12), B.randn(tf.float32, 16, 1, 12)),
            B.randn(tf.float32, 16, dim_y2, 12, 12),
        )
    ],
    B.randn(tf.float32, 16, dim_x, 15),
    aux_t=B.randn(tf.float32, 16, dim_aux_t, 15),
)
```
# Build Your Own Model

NeuralProcesses offers building blocks which can be put together in various ways to
construct models suited to a particular application.

## Examples in PyTorch

None yet.

## Examples in TensorFlow

## ConvGNP

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
        nps.SetConv(scale=1 / disc.points_per_unit),
        nps.DivideByFirstChannel(),
        nps.DeterministicLikelihood(),
    ),
)
decoder = nps.Chain(
    unet,
    nps.SetConv(scale=1 / disc.points_per_unit),
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

## ConvGNP with Auxiliary Variables

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
        # scales of these set convs both to `1 / disc.points_per_unit`.
        nps.Parallel(
            nps.SetConv(scale=1 / disc.points_per_unit),
            nps.SetConv(scale=1 / disc.points_per_unit),
        ),
        nps.DivideByFirstChannel(),
        # Concatenate the encodings of the context sets.
        nps.Concatenate(),
        nps.DeterministicLikelihood(),
    ),
)
decoder = nps.Chain(
    unet,
    nps.SetConv(scale=1 / disc.points_per_unit),
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

# Basic Usage

## Backend Agnostic

NeuralProcesses is designed to work with both PyTorch and TensorFlow.
For use with PyTorch, import as

```python
import neuralprocesses.torch as nps
```

For use with TensorFlow, import as

```python
import neuralprocesses.tensorflow as nps
```

We plan to also support JAX.

## Shape of Tensors

Inputs and outputs are always tensors of shape `(b, c, n)` where `b` is the
batch size, `c` is the dimensionality of the inputs/outputs or the number of channels,
and `n` is the number of data points.
This convention is extended in two ways.

First, the batch size `b` can have multiple dimensions.
For example, a single sample from a model will have shape `(b, c, n)`,
but `s > 1` samples will have shape `(s, b, c, n)`.
Here we interpret the extra dimension `s` as an extra batch dimension
and write the shape as `(*b, c, n)` where `b = (b1, b2)`.

Second, the number of data points `n` can have multiple dimensions.
For example, if we are dealing with images, which have two dimensions, tensors
will have shape `(b, c, n1, n2)` where the shape of the images is `(n1, n2)`.
Here we intepret the shape of the image as two `n` dimensions and write the
shape as `(b, c, *n)` where `n = (n1, n2)`.

In conclusion, tensors are always of shape `(*b, c, *n)`
where `b` are one or multiple batch dimensions,
`c` is the dimensionality of the inputs/output or the number of channels,
and `n` are one or multiple data dimensions.


## Examples for PyTorch

### GNP

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

### ConvGNP

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

## Examples for TensorFlow

### GNP

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

### ConvGNP

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

### ConvGNP with Auxiliary Variables

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
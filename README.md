# [Neural Processes](http://github.com/wesselb/neuralprocesses)

[![CI](https://github.com/wesselb/neuralprocesses/workflows/CI/badge.svg)](https://github.com/wesselb/neuralprocesses/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/neuralprocesses/badge.svg)](https://coveralls.io/github/wesselb/neuralprocesses?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/neuralprocesses)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A framework for composing Neural Processes in Python.
See also [NeuralProcesses.jl](https://github.com/wesselb/NeuralProcesses.jl).

*This package is currently under construction.
There will be more here soon. In the meantime, see
[NeuralProcesses.jl](https://github.com/wesselb/NeuralProcesses.jl).*

## Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```
pip install neuralprocesses
```

## Example

### TensorFlow

#### CNP

```python
import lab as B
import tensorflow as tf

import neuralprocesses.tensorflow as nps

cnp = nps.construct_cnp(dim_x=2, dim_y=3)
dist = cnp(
    B.randn(tf.float32, 16, 2, 10),
    B.randn(tf.float32, 16, 3, 10),
    B.randn(tf.float32, 16, 2, 15),
)
print(dist.logpdf(B.randn(tf.float32, 16, 3, 15)))
print(dist.kl(dist))
print(dist.entropy())
print(dist.mean)
print(dist.var)
```

#### ConvGNP

```python
import lab as B
import tensorflow as tf

import neuralprocesses.tensorflow as nps

cnp = nps.construct_convgnp(dim_x=2, dim_y=3, likelihood="lowrank")

dist = cnp(
    B.randn(tf.float32, 16, 2, 10),
    B.randn(tf.float32, 16, 3, 10),
    B.randn(tf.float32, 16, 2, 15),
)
mean, var = dist.mean, dist.var

print(dist.logpdf(B.randn(tf.float32, 16, 3, 15)))
print(dist.kl(dist))
print(dist.entropy())
```

### PyTorch

#### CNP

```python
import lab as B
import torch

import neuralprocesses.torch as nps

cnp = nps.construct_cnp(dim_x=2, dim_y=3)
dist = cnp(
    B.randn(torch.float32, 16, 2, 10),
    B.randn(torch.float32, 16, 3, 10),
    B.randn(torch.float32, 16, 2, 15),
)
mean, var = dist.mean, dist.var

print(dist.logpdf(B.randn(torch.float32, 16, 3, 15)))
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
    B.randn(torch.float32, 16, 2, 10),
    B.randn(torch.float32, 16, 3, 10),
    B.randn(torch.float32, 16, 2, 15),
)
mean, var = dist.mean, dist.var

print(dist.logpdf(B.randn(torch.float32, 16, 3, 15)))
print(dist.kl(dist))
print(dist.entropy())
```
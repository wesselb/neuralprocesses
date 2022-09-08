# Advanced Usage

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

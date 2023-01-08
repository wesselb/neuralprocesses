from functools import wraps
from string import ascii_lowercase as letters

import lab as B

from ... import _dispatch
from ...augment import AugmentedInput
from ...parallel import broadcast_coder_over_parallel
from ...util import register_module

from . import privacy_accounting as pa

from stheno import EQ

__all__ = ["SetConv", "DPSetConv"]


@register_module
class SetConv:
    """A set convolution.
    Args:
        scale (float): Initial value for the length scale.
        dtype (dtype, optional): Data type.
        learnable (bool, optional): Whether the SetConv length scale is learnable.
    Attributes:
        log_scale (scalar): Logarithm of the length scale.
    """

    def __init__(self, scale, dtype=None, learnable=True):
        self.log_scale = self.nn.Parameter(
            B.log(scale), dtype=dtype, learnable=learnable
        )


def _dim_is_concrete(x, i):
    try:
        int(B.shape(x, i))
        return True
    except TypeError:
        return False


def _batch_targets(f):
    @wraps(f)
    def f_wrapped(coder, xz, z, x, batch_size=1024, **kw_args):
        # If `x` is the internal discretisation and we're compiling this function
        # with `tf.function`, then `B.shape(x, -1)` will be `None`. We therefore
        # check that `B.shape(x, -1)` is concrete before attempting the comparison.
        if _dim_is_concrete(x, -1) and B.shape(x, -1) > batch_size:
            i = 0
            outs = []
            while i < B.shape(x, -1):
                outs.append(
                    code(
                        coder,
                        xz,
                        z,
                        x[..., i : i + batch_size],
                        batch_size=batch_size,
                        **kw_args,
                    )[1]
                )
                i += batch_size
            return x, B.concat(*outs, axis=-1)
        else:
            return f(coder, xz, z, x, **kw_args)

    return f_wrapped


def compute_weights(coder, x1, x2):
    # Compute interpolation weights.
    dists2 = B.pw_dists2(B.transpose(x1), B.transpose(x2))
    return B.exp(-0.5 * dists2 / B.exp(2 * coder.log_scale))


@_dispatch
@_batch_targets
def code(coder: SetConv, xz: B.Numeric, z: B.Numeric, x: B.Numeric, **kw_args):
    return x, B.matmul(z, compute_weights(coder, xz, x))


_setconv_cache_num_tup = {}


@_dispatch
def code(coder: SetConv, xz: B.Numeric, z: B.Numeric, x: tuple, **kw_args):
    ws = [compute_weights(coder, xz[..., i : i + 1, :], xi) for i, xi in enumerate(x)]

    # Use a cache so we don't build the equation every time.
    try:
        equation = _setconv_cache_num_tup[len(x)]
    except KeyError:
        letters_i = 3
        base = "...bc"
        result = "...b"
        for _ in range(len(x)):
            let = letters[letters_i]
            letters_i += 1
            base += f",...c{let}"
            result += f"{let}"
        _setconv_cache_num_tup[len(x)] = f"{base}->{result}"
        equation = _setconv_cache_num_tup[len(x)]

    return x, B.einsum(equation, z, *ws)


_setconv_cache_tup_num = {}


@_dispatch
@_batch_targets
def code(coder: SetConv, xz: tuple, z: B.Numeric, x: B.Numeric, **kw_args):
    ws = [compute_weights(coder, xzi, x[..., i : i + 1, :]) for i, xzi in enumerate(xz)]

    # Use a cache so we don't build the equation every time.
    try:
        equation = _setconv_cache_tup_num[len(xz)]
    except KeyError:
        letters_i = 3
        base_base = "...b"
        base_els = ""
        for _ in range(len(xz)):
            let = letters[letters_i]
            letters_i += 1
            base_base += f"{let}"
            base_els += f",...{let}c"
        _setconv_cache_tup_num[len(xz)] = f"{base_base}{base_els}->...bc"
        equation = _setconv_cache_tup_num[len(xz)]

    return x, B.einsum(equation, z, *ws)


_setconv_cache_tup_tup = {}


@_dispatch
def code(coder: SetConv, xz: tuple, z: B.Numeric, x: tuple, **kw_args):
    ws = [compute_weights(coder, xzi, xi) for xzi, xi in zip(xz, x)]

    # Use a cache so we don't build the equation every time.
    try:
        equation = _setconv_cache_tup_tup[len(x)]
    except KeyError:
        letters_i = 2
        base_base = "...b"
        base_els = ""
        result = "...b"
        for _ in range(len(x)):
            let1 = letters[letters_i]
            letters_i += 1
            let2 = letters[letters_i]
            letters_i += 1
            base_base += f"{let1}"
            base_els += f",...{let1}{let2}"
            result += f"{let2}"
        _setconv_cache_tup_tup[len(x)] = f"{base_base}{base_els}->{result}"
        equation = _setconv_cache_tup_tup[len(x)]

    return x, B.einsum(equation, z, *ws)


broadcast_coder_over_parallel(SetConv)


@_dispatch
def code(coder: SetConv, xz, z, x: AugmentedInput, **kw_args):
    xz, z = code(coder, xz, z, x.x, **kw_args)
    return AugmentedInput(xz, x.augmentation), z


@register_module
class DPSetConv:
    """A set convolution with a DP mechanism.

    Args:
        scale (float): Initial value for the length scale.
        dtype (dtype, optional): Data type.
        learnable (bool, optional): Whether the SetConv length scale is learnable.

    Attributes:
        log_scale (scalar): Logarithm of the length scale.

    """

    def __init__(
            self,
            scale,
            y_bound,
            epsilon,
            delta,
            dtype=None,
            learnable=True,
        ):
        
        self.epsilon = epsilon
        self.delta = delta
        
        self.log_scale = self.nn.Parameter(
            B.log(scale), dtype=dtype, learnable=learnable
        )
        
        self.log_y_bound = self.nn.Parameter(
            B.log(y_bound), dtype=dtype, learnable=learnable
        )
        
        self.sens_per_sigma = pa.find_sens_per_sigma(epsilon=epsilon, delta_bound=delta)
    
    @property
    def density_sigma(self):
        return 2.0**0.5 / self.sens_per_sigma
    
    @property
    def value_sigma(self):
        return 2 * B.exp(self.log_y_bound) / self.sens_per_sigma
      
    @property
    def y_bound(self):
        return B.exp(self.log_y_bound)
        
    def sample_noise(self, xz, z, x):
        
        _x = B.transpose(x, [0, 2, 1])
        
        kernel = EQ().stretch(B.exp(self.log_scale))
        
        k = lambda tensor: kernel(tensor, tensor) + \
            1e-6 * B.eye(tensor.dtype, tensor.shape[-1])[None, :, :]
        
        noise = [B.sample(k(_x.double()))[:, None, :, 0] for i in range(z.shape[1])]
        noise = B.cast(z.dtype, B.concat(*noise, axis=1))
        
        num_channels = noise.shape[1]
        
        noise_density = self.density_sigma * noise[:, :num_channels//2, :]
        noise_value = self.value_sigma * noise[:, num_channels//2:, :]
        
        noise = B.concat(noise_density, noise_value, axis=1)
    
        return noise
    
    def clip_data_channel(self, tensor):
        
        num_channels = tensor.shape[1]
        ones = B.ones(tensor.dtype, *(tensor[:, :num_channels//2, :].shape))

        kernel = tensor[:, :num_channels//2, :]

        clipped_data = B.minimum(tensor[:, num_channels//2:, :], self.y_bound * ones)
        clipped_data = B.maximum(clipped_data, -self.y_bound * ones)
        
        tensor = B.concat(kernel, clipped_data, axis=1)
        
        return tensor

        
@_dispatch
@_batch_targets
def code(coder: DPSetConv, xz: B.Numeric, z: B.Numeric, x: B.Numeric, **kw_args):
    
    density_data_channels = B.matmul(
        coder.clip_data_channel(z),
        compute_weights(coder, xz, x),
    )
    
    z = density_data_channels + coder.sample_noise(xz, z, x)
    
    return x, z


_setconv_cache_num_tup = {}

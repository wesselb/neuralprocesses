from functools import wraps
from string import ascii_lowercase as letters

import lab as B
import torch

from ... import _dispatch
from ...augment import AugmentedInput
from ...parallel import broadcast_coder_over_parallel
from ...util import register_module

from torchvision.ops import MLP
from . import privacy_accounting as pa

from stheno import EQ

import gpytorch

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

    def __init__(
        self,
        scale,
        y_bound,
        t,
        amortise_dp_params,
        learnable_scale,
        learnable_dp_params,
        dtype,
    ):
        """Initialise DPSetConv, a set convolution with a DP mechanism.

        Arguments:
            scale (float): Initial value for the length scale.
            y_bound (float): Initial value for the y-bound DP parameter.
            t (float): Initial value for the t DP parameter.
            amortise_dp_params (bool): Whether to amortise the DP parameters.
            learnable_scale (bool): Whether the SetConv length scale is learnable.
            learnable_dp_params (bool): Whether the DP parameters are learnable.
            dtype (dtype, optional): Data type for the DPSetConv.
        """

        # Whether to amortise the y_bound and t DP parameters
        self.amortise_dp_params = amortise_dp_params

        # If amortising the DP parameeters, use a small MLP to learn them
        if self.amortise_dp_params:

            assert learnable_dp_params
    
            self.y_mlp = MLP(
                in_channels=1,
                hidden_channels=[20, 1],
            )
    
            self.t_mlp = MLP(
                in_channels=1,
                hidden_channels=[20, 1],
            )

        # If not amortising the DP parameters, use a single parameter for each,
        # set them to the given values and specify whether they are learnable
        else:

            self.log_y_bound = self.nn.Parameter(
                B.log(y_bound), dtype=dtype, learnable=learnable_dp_params,
            )

            self.logit_t = self.nn.Parameter(
                torch.logit(t), dtype=dtype, learnable=learnable_dp_params,
            )

        # Initialise the log-scale and specify whether it is learnable
        self.log_scale = self.nn.Parameter(
            B.log(scale), dtype=dtype, learnable=learnable_scale,
        )

    def density_sigma(self, sens_per_sigma):
        """Compuete the density noise sigma for a given sensitivity per sigma.

        Arguments:
            sens_per_sigma (torch.Tensor): Sensitivity per sigma.

        Returns:
            torch.Tensor: Density channel noise sigma.
        """
        return 2**0.5 / (sens_per_sigma * (1 - self.t(sens_per_sigma)) ** 0.5)

    def data_sigma(self, sens_per_sigma):
        """Compuete the data noise sigma for a given sensitivity per sigma.

        Arguments:
            sens_per_sigma (torch.Tensor): Sensitivity per sigma.

        Returns:
            torch.Tensor: Data channel noise sigma.
        """
        return (
            2.0
            * self.y_bound(sens_per_sigma)
            / (sens_per_sigma * self.t(sens_per_sigma) ** 0.5)
        )

    # Correct:
    #     both_sigma = (2 + 4 * y_bound**2)**0.5 / sens_per_sigma
    #
    # Also correct:
    #
    #     t in [0, 1]
    #
    #     μ = sens_per_sigma ** 2 / 2
    #     density_sensitivity = 2 ** 0.5
    #     signal_sensitivity = 2 * y_bound
    #
    #     density_sigma = density_sensitivity / (2 (1 - t) μ)**0.5 = density_sens / (sens_per_sigma * (1 - t)**0.5)
    #     data_sigma = signal_sensitivity / (2 t μ)**0.5 = signal_sens / (sens_per_sigma * t**0.5)
    #
    #     density_sigma = 2**0.5 * density_sensitivity / sens_per_sigma = 2 / sens_per_sigma
    #     data_sigma = 2**0.5 * signal_sensitivity / sens_per_sigma = 2 * 2**0.5 * y_bound

    def t(self, sens_per_sigma):
        t = B.sigmoid(
            self.t_mlp(sens_per_sigma[:, None])[:, 0]
        ) if self.amortise_dp_params else B.sigmoid(self.logit_t[None])

        return 1e-2 + (1 - 2e-2) * t

    def y_bound(self, sens_per_sigma):
        y_bound = B.exp(
            self.y_mlp(sens_per_sigma[:, None])[:, 0]
        ) if self.amortise_dp_params else B.exp(self.log_y_bound[None])

        return 1e-3 + y_bound

    def sample_noise(self, z, x, sens_per_sigma):
        _x = B.transpose(x, [0, 2, 1])

        torch.set_default_dtype(torch.float64)

        kernel = gpytorch.kernels.RBFKernel().to(z.device)
        kernel.lengthscale = self.log_scale.exp().double()

        kxx = kernel(_x.double()) + 1e-6 * B.eye(_x.dtype, _x.shape[-1])[None, :, :]
        p_noise = gpytorch.distributions.MultivariateNormal(
            mean=torch.zeros(*kxx.shape[:-1], device=kxx.device),
            covariance_matrix=kxx,
        )
        noise = [p_noise.rsample()[:, None, :] for _ in range(z.shape[1])]
        noise = B.cast(z.dtype, B.concat(*noise, axis=1))

        torch.set_default_dtype(z.dtype)

        num_channels = noise.shape[1]

        density_noise = self.density_sigma(sens_per_sigma)[:, None, None] * noise[:, : num_channels // 2, :]
        data_noise = self.data_sigma(sens_per_sigma)[:, None, None] * noise[:, num_channels // 2 :, :]

        noise = B.concat(density_noise, data_noise, axis=1)

        return noise

    def clip_data_channel(self, tensor, sens_per_sigma):
        num_channels = tensor.shape[1]
        ones = B.ones(tensor.dtype, *(tensor[:, : num_channels // 2, :].shape))

        kernel = tensor[:, : num_channels // 2, :]

        clipped_data = B.minimum(
            tensor[:, num_channels // 2 :, :],
            self.y_bound(sens_per_sigma)[:, None, None] * ones,
        )
        clipped_data = B.maximum(
            clipped_data, -self.y_bound(sens_per_sigma)[:, None, None] * ones
        )

        tensor = B.concat(kernel, clipped_data, axis=1)

        return tensor

    def sens_per_sigma(self, epsilon, delta):
        sens_per_sigma = torch.tensor(
            [
                pa.find_sens_per_sigma(epsilon=e, delta_bound=d)
                for e, d in zip(epsilon[:, 0], delta[:, 0])
            ]
        )

        return sens_per_sigma


@_dispatch
@_batch_targets
def code(
    coder: DPSetConv,
    xz: B.Numeric,
    z: B.Numeric,
    x: B.Numeric,
    *,
    epsilon: B.Numeric,
    delta: B.Numeric,
    **kw_args,
):
    sens_per_sigma = coder.sens_per_sigma(epsilon, delta).to(
        xz.device
    )  # shape: (batch_size,)

    density_data_channels = B.matmul(
        coder.clip_data_channel(z, sens_per_sigma),
        compute_weights(coder, xz, x),
    )

    z = density_data_channels + coder.sample_noise(z, x, sens_per_sigma)

    return x, z


_setconv_cache_num_tup = {}

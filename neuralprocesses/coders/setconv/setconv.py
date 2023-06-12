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
        dp_learn_params,
        dp_amortise_params,
        learnable_scale,
        dp_use_noise_channels,
        dtype,
    ):
        """Initialise DPSetConv, a set convolution with a DP mechanism.

        Arguments:
            scale (float): Initial value for the length scale.
            y_bound (float): Initial value for the y-bound DP parameter.
            t (float): Initial value for the t DP parameter.
            dp_learn_params (bool): Whether to learn the DP parameters.
            dp_amortise_params (bool): Whether to amortise the DP parameters.
            learnable_scale (bool): Whether the length scale is learnable.
            dp_use_noise_channels (bool): Whether to append noise std to output.
            dtype (dtype, optional): Data type for the DPSetConv.
        """

        # Whether to amortise the y_bound and t DP parameters
        self.dp_amortise_params = dp_amortise_params

        # Whether to append the density and data channel noise std to the output
        self.dp_use_noise_channels = dp_use_noise_channels

        # If amortising the DP parameeters, use a small MLP to learn them
        if self.dp_amortise_params:

            self.y_mlp = MLP(
                in_channels=1,
                hidden_channels=[20, 20, 1],
            )
    
            self.t_mlp = MLP(
                in_channels=1,
                hidden_channels=[20, 20, 1],
            )

        # If not amortising the DP parameters, use a single parameter for each,
        # set them to the given values and specify whether they are learnable
        else:

            assert y_bound is not None and t is not None, (
                f"Must specify y_bound and t if not amortising DP parameters, "
                f"got {y_bound=} and {t=}."
            )

            self.log_y_bound = self.nn.Parameter(
                B.log(torch.tensor(y_bound)), dtype=dtype, learnable=dp_learn_params,
            )

            self.logit_t = self.nn.Parameter(
                torch.logit(torch.tensor(t)), dtype=dtype, learnable=dp_learn_params,
            )

        # Initialise the log-scale and specify whether it is learnable
        self._log_scale = self.nn.Parameter(
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
        """Compute the t DP parameter for a given sensitivity per sigma. The
        t parameter determines how the total noise level is split between the
        density and data channels.

        Arguments:
            sens_per_sigma (torch.Tensor): Sensitivity per sigma.

        Returns:
            torch.Tensor: t DP parameter.
        """

        # Compute the t parameter: if this is amortised, it is a function of
        # the sensitivity per sigma, otherwise it is a constant, and the
        # sensitivity per sigma is ignored
        t = B.sigmoid(
            self.t_mlp(sens_per_sigma[:, None])[:, 0]
        ) if self.dp_amortise_params else B.sigmoid(self.logit_t[None])

        return 1e-2 + (1 - 2e-2) * t


    def y_bound(self, sens_per_sigma):
        """Compute the y-bound DP parameter for a given sensitivity per sigma.
        The y-bound parameter determines the maximum value of the data channel,
        i.e. it is threshold value with which the datapoint values are clipped.

        Arguments:
            sens_per_sigma (torch.Tensor): Sensitivity per sigma.

        Returns:
            torch.Tensor: y-bound DP parameter.
        """

        # Compute the y-bound parameter: if this is amortised, it is a function
        # of the sensitivity per sigma, otherwise it is a constant, and the
        # sensitivity per sigma is ignored
        y_bound = B.exp(
            self.y_mlp(sens_per_sigma[:, None])[:, 0]
        ) if self.dp_amortise_params else B.exp(self.log_y_bound[None])

        return 1e-3 + y_bound


    @property
    def log_scale(self):
        """Return the length scale of the SetConv."""

        return torch.log(self._log_scale.exp() + 2. / 32.)


    def sample_noise(self, x, sens_per_sigma, num_channels):
        """Sample EQ-GP noise for the density and data channels. The lengthscale
        of the EQ-GP is the same as the scale of the EQ basis function used in
        the density and data channels.

        Arguments:
            x (torch.Tensor): Input to the SetConv.
            sens_per_sigma (torch.Tensor): Sensitivity per sigma.
            num_channels (int): Number of channels (density + data channels).

        Returns:
            torch.Tensor: Sampled noise.
        """

        ## Transpose x so that the x-dimensions are the last dimensions
        #xT = B.transpose(x, [0, 2, 1])

        ## Use double precision for sampling the noise
        #torch.set_default_dtype(torch.float64)

        ## Initialise the EQ kernel and set its lengthscale
        #kernel = gpytorch.kernels.RBFKernel().to(x.device)
        #kernel.lengthscale = self.log_scale.exp().double()

        # Compute the covariance matrix of the EQ-GP
        #kxx = kernel(xT.double())

        kxx = torch.exp(
            -0.5 * torch.sum(
                (x[:, :, None, :].double() - x[:, :, :, None].double())**2.,
                dim=1,
            ) / torch.exp(self.log_scale).double()**2.
        ) 

        kxx = kxx + 1e-3 * B.eye(
            x.dtype,
            x.shape[-1],
        )[None, :, :]

        kxx_chol = torch.linalg.cholesky(kxx)

        ## Set noise distribution
        #p_noise = gpytorch.distributions.MultivariateNormal(
        #    mean=torch.zeros(*kxx.shape[:-1], device=kxx.device),
        #    covariance_matrix=kxx,
        #)

        ## Sample noise for each channel separately and concatenate
        #noise = [p_noise.rsample()[:, None, :] for _ in range(num_channels)]
        #noise = B.cast(x.dtype, B.concat(*noise, axis=1))
        noise_shape = (x.shape[0], num_channels, x.shape[2])
        noise = torch.randn(*noise_shape, device=x.device, dtype=kxx.dtype)
        noise = torch.einsum("bnm, bcm -> bcn", kxx_chol, noise).float()

        ## Reset the default dtype to the original setting
        #torch.set_default_dtype(x.dtype)

        # Compute the noise sigma for the density and data channels
        density_sigma = self.density_sigma(
            sens_per_sigma=sens_per_sigma,
        )[:, None, None]
        density_noise = density_sigma * noise[:, : num_channels // 2, :]

        data_sigma = self.data_sigma(
            sens_per_sigma=sens_per_sigma,
        )[:, None, None]
        data_noise = data_sigma * noise[:, num_channels // 2 :, :]

        # Concatenate the density and data channel noise
        noise = B.concat(density_noise, data_noise, axis=1)

        return noise, density_sigma, data_sigma


    def clip_data(self, z, sens_per_sigma):
        """Clip the data channel values to the y-bound DP parameter.

        Arguments:
            tensor (torch.Tensor): Input to the SetConv.
        """

        # Get the number of channels
        num_channels = z.shape[1]

        # Compute clipping threshold y_bound: if this is amortised, it is a
        # function of the sensitivity per sigma, otherwise it is a constant,
        # and the sensitivity per sigma is ignored
        threshold = self.y_bound(
            sens_per_sigma=sens_per_sigma,
        )[:, None, None] * B.ones(z.dtype, *(z[:, : num_channels // 2, :].shape))

        # Threshold data from aboeve and then from below
        clipped_data = B.minimum(z[:, num_channels // 2 :, :], threshold)
        clipped_data = B.maximum(clipped_data, -threshold)
        
        # Concatenate the density with the clipped data
        z = B.concat(z[:, : num_channels // 2, :], clipped_data, axis=1)

        return z

    def sens_per_sigma(self, epsilon, delta):
        """Compute the sensitivity per sigma for a given epsilon and delta,
        using the privacy accounting module.

        Arguments:
            epsilon (torch.Tensor): Epsilon.
            delta (torch.Tensor): Delta.

        Returns:
            torch.Tensor: Sensitivity per sigma.
        """

        # Compute the sensitivity per sigma for each task in the batch
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

    # Compute the sensitivity per sigma for the given epsilon and delta.
    # Note: each task in the batch may have different epsilon and delta.
    sens_per_sigma = coder.sens_per_sigma(epsilon, delta).to(
        xz.device
    )  # shape: (batch_size,)

    # Clip the data channel values to the y-bound DP parameter, and apply
    # the SetConv to the clipped data.
    z = B.matmul(
        z, # coder.clip_data(z, sens_per_sigma),
        compute_weights(coder, xz, x),
    )

    # Sample noise for the density and data channels
    noise, density_sigma, data_sigma =  coder.sample_noise(
        x=x,
        sens_per_sigma=sens_per_sigma,
        num_channels=z.shape[1],
    )

    # Add noise to the density and data channels
    z = z + noise

    # If specified, concatenate density and data sigmas to the output
    if coder.dp_use_noise_channels:

        # Broadcast density and data sigmas to the same shape as z, except
        # for the last dimension, which is one for the density and data
        density_sigma = density_sigma * B.ones(z.dtype, z.shape[0], 1, z.shape[2])
        data_sigma = data_sigma * B.ones(z.dtype, z.shape[0], 1, z.shape[2])

        # Concatenate density and data sigmas to the output
        z = B.concat(z, density_sigma, data_sigma, axis=1)

    return x, z


_setconv_cache_num_tup = {}

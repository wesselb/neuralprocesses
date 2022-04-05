import wbml.out as out
from plum import convert

import neuralprocesses as nps  # This fixes inspection below.
from .lik import construct_likelihood
from ..util import register_model

__all__ = ["construct_convgnp"]


@register_model
def construct_convgnp(
    dim_x=1,
    dim_y=1,
    dim_yc=None,
    dim_yt=None,
    dim_aux_t=None,
    points_per_unit=64,
    margin=0.1,
    likelihood="lowrank",
    conv_arch="unet",
    unet_channels=(64,) * 6,
    unet_kernels=5,
    unet_activations=None,
    unet_resize_convs=False,
    unet_resize_conv_interp_method="nearest",
    dws_receptive_field=None,
    dws_layers=8,
    dws_channels=64,
    num_basis_functions=512,
    encoder_scales=None,
    decoder_scale=None,
    aux_t_mlp_layers=(128,) * 3,
    epsilon=1e-4,
    transform=None,
    dtype=None,
    nps=nps,
):
    """A Convolutional Gaussian Neural Process.

    Sets the attribute `receptive_field` to the receptive field of the model.

    Args:
        dim_x (int, optional): Dimensionality of the inputs. Defaults to `1`.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to `1`.
        dim_yc (int or tuple[int], optional): Dimensionality of the outputs of the
            context set. You should set this if the dimensionality of the outputs
            of the context set is not equal to the dimensionality of the outputs
            of the target set. You should also set this if you want to use multiple
            context sets. In that case, set this equal to a tuple of integers
            indicating the respective output dimensionalities.
        dim_yt (int, optional): Dimensionality of the outputs of the target set. You
            should set this if the dimensionality of the outputs of the target set is
            not equal to the dimensionality of the outputs of the context set.
        dim_aux_t (int, optional): Dimensionality of target-specific auxiliary
            variables.
        points_per_unit (float, optional): Density of the internal discretisation.
            Defaults to `64`.
        margin (float, optional): Margin of the internal discretisation. Defaults to
            `0.1`
        likelihood (str, optional): Likelihood. Must be one of "het", "lowrank", or
            "lowrank-correlated". Defaults to "lowrank".
        conv_arch (str, optional): Convolutional architecture to use. Must be one of
            "unet" or "dws". Defaults to "unet.
        unet_channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        unet_kernels (int or tuple[int], optional): Sizes of the kernels in the UNet.
            Defaults to `5`.
        unet_activations (object or tuple[object], optional): Activation functions
            used by the UNet.
        unet_resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions in the UNet. Defaults to `False`.
        unet_resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions in the UNet. Can be set to "bilinear". Defaults
            to "nearest".
        dws_receptive_field (float, optional): Receptive field of the DWS architecture.
            Must be specified if `conv_arch` is set to "dws".
        dws_layers (int, optional): Layers of the DWS architecture. Defaults to `8`.
        dws_channels (int, optional): Channels of the DWS architecture. Defaults to
            `64`.
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to `512`.
        encoder_scales (float or tuple[float], optional): Initial value for the length
            scales of the set convolutions for the context sets embeddings. Defaults
            to `2 / points_per_unit`.
        decoder_scale (float, optional): Initial value for the length scale of the
            set convolution in the decoder. Defaults to `2 / points_per_unit`.
        aux_t_mlp_layers (tuple[int], optional): Widths of the layers of the MLP
            for the target-specific auxiliary variable. Defaults to three layers of
            width 128.
        epsilon (float, optional): Epsilon added by the set convolutions before
            dividing by the density channel. Defaults to `1e-4`.
        transform (str or tuple[float, float], optional): Bijection applied to the
            output of the ConvGNP. This can help deal with positive of bounded data.
            Must be either "positive" for positive data or `(lower, upper)` for data
            in this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: ConvGNP model.
    """
    # Make sure that `dim_yc` is initialised and a tuple.
    dim_yc = convert(dim_yc or dim_y, tuple)
    # Make sure that `dim_yt` is initialised.
    dim_yt = dim_yt or dim_y
    # `len(dim_yc)` is equal to the number of density channels.
    conv_in_channels = sum(dim_yc) + len(dim_yc)
    likelihood_in_channels, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=dim_yt,
        num_basis_functions=num_basis_functions,
        dtype=dtype,
    )

    # Resolve architecture.
    if conv_arch == "unet":
        conv_out_channels = unet_channels[0]
    elif conv_arch == "dws":
        conv_out_channels = dws_channels

        if dws_receptive_field is None:
            raise ValueError(f"Must specify `dws_receptive_field`.")
    else:
        raise ValueError(f'Architecture "{conv_arch}" invalid.')

    # If `dim_aux_t` is given, contruct an MLP which will use the auxiliary
    # information from the augmented inputs.
    if dim_aux_t:
        likelihood = nps.Augment(
            nps.Chain(
                nps.MLP(
                    in_dim=conv_out_channels + dim_aux_t,
                    layers=aux_t_mlp_layers,
                    out_dim=likelihood_in_channels,
                    dtype=dtype,
                ),
                likelihood,
            )
        )
        likelihood_in_channels = unet_channels[0]

    # If `transform` is set to a value, apply the transform.
    if isinstance(transform, str) and transform.lower() == "positive":
        likelihood = nps.Chain(likelihood, nps.Transform.positive())
    elif isinstance(transform, tuple):
        lower, upper = transform
        likelihood = nps.Chain(likelihood, nps.Transform.bounded(lower, upper))
    elif transform is not None:
        raise ValueError(f'Cannot parse value "{transform}" for `transform`.')

    # Construct the core CNN architecture of the model.
    if conv_arch == "unet":
        conv = nps.UNet(
            dim=dim_x,
            in_channels=conv_in_channels,
            out_channels=likelihood_in_channels,
            channels=unet_channels,
            kernels=unet_kernels,
            activations=unet_activations,
            resize_convs=unet_resize_convs,
            resize_conv_interp_method=unet_resize_conv_interp_method,
            dtype=dtype,
        )
        receptive_field = conv.receptive_field / points_per_unit
    elif conv_arch == "dws":
        conv = nps.ConvNet(
            dim=dim_x,
            in_channels=conv_in_channels,
            out_channels=likelihood_in_channels,
            channels=dws_channels,
            num_layers=dws_layers,
            points_per_unit=points_per_unit,
            receptive_field=dws_receptive_field,
            dtype=dtype,
        )
        receptive_field = dws_receptive_field
    else:
        raise ValueError(f'Architecture "{conv_arch}" invalid.')

    # Construct the discretisation, taking into account that the input to the UNet
    # must play nice with the halving layers.
    disc = nps.Discretisation(
        points_per_unit=points_per_unit,
        multiple=2**conv.num_halving_layers,
        margin=margin,
        dim=dim_x,
    )

    # Construct a separate set conv for every context set.
    encoder_scales = encoder_scales or 2 / disc.points_per_unit
    if not isinstance(encoder_scales, (tuple, list)):
        encoder_scales = (encoder_scales,) * len(dim_yc)
    encoder_set_conv = nps.Parallel(
        *(nps.SetConv(s, dtype=dtype) for s in encoder_scales)
    )

    # Resolve length scale for decoder.
    decoder_scale = decoder_scale or 2 / disc.points_per_unit

    # Construct model.
    model = nps.Model(
        nps.FunctionalCoder(
            disc,
            nps.Chain(
                nps.PrependDensityChannel(),
                encoder_set_conv,
                nps.DivideByFirstChannel(epsilon=epsilon),
                nps.Materialise(),
            ),
        ),
        nps.Chain(
            conv,
            nps.SetConv(decoder_scale, dtype=dtype),
            likelihood,
        ),
    )

    # Set attribute `receptive_field`.
    out.kv("Receptive field", receptive_field)
    model.receptive_field = receptive_field

    return model

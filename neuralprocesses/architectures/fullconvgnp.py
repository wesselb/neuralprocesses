import lab as B
import wbml.out as out
from plum import convert

import neuralprocesses as nps  # This fixes inspection below.
from ..util import register_model

__all__ = ["construct_fullconvgnp"]


@register_model
def construct_fullconvgnp(
    dim_x=1,
    dim_y=1,
    dim_yc=None,
    dim_yt=None,
    points_per_unit=64,
    margin=0.1,
    conv_arch="unet",
    unet_channels=(64,) * 6,
    unet_kernels=5,
    unet_activations=None,
    unet_resize_convs=False,
    unet_resize_conv_interp_method="nearest",
    dws_receptive_field=None,
    dws_layers=8,
    dws_channels=64,
    encoder_scales=None,
    decoder_scale=None,
    epsilon=1e-4,
    transform=None,
    dtype=None,
    nps=nps,
):
    """A Fully Convolutional Gaussian Neural Process.

    Sets the attribute `receptive_field` to the receptive field of the model.

    Args:
        dim_x (int, optional): Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional): Dimensionality of the outputs of the
            context set. You should set this if the dimensionality of the outputs
            of the context set is not equal to the dimensionality of the outputs
            of the target set. You should also set this if you want to use multiple
            context sets. In that case, set this equal to a tuple of integers
            indicating the respective output dimensionalities.
        dim_yt (int, optional): Dimensionality of the outputs of the target set. You
            should set this if the dimensionality of the outputs of the target set is
            not equal to the dimensionality of the outputs of the context set.
        points_per_unit (float, optional): Density of the internal discretisation.
            Defaults to 64.
        margin (float, optional): Margin of the internal discretisation. Defaults to
            0.1.
        conv_arch (str, optional): Convolutional architecture to use. Must be one of
            `"unet"` or `"dws"`. Defaults to `"unet"`.
        unet_channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        unet_kernels (int or tuple[int], optional): Sizes of the kernels in the UNet.
            Defaults to 5.
        unet_activations (object or tuple[object], optional): Activation functions
            used by the UNet.
        unet_resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions in the UNet. Defaults to `False`.
        unet_resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions in the UNet. Can be set to "bilinear". Defaults
            to "nearest".
        dws_receptive_field (float, optional): Receptive field of the DWS architecture.
            Must be specified if `conv_arch` is set to "dws".
        dws_layers (int, optional): Layers of the DWS architecture. Defaults to 8.
        dws_channels (int, optional): Channels of the DWS architecture. Defaults to 64.
        encoder_scales (float or tuple[float], optional): Initial value for the length
            scales of the set convolutions for the context sets embeddings. Defaults
            to `2 / points_per_unit`.
        decoder_scale (float, optional): Initial value for the length scale of the
            set convolution in the decoder. Defaults to `2 / points_per_unit`.
        epsilon (float, optional): Epsilon added by the set convolutions before
            dividing by the density channel. Defaults to `1e-4`.
        transform (str or tuple[float, float], optional): Bijection applied to the
            output of the ConvGNP. This can help deal with positive of bounded data.
            Must be either "positive" for positive data or `(lower, upper)` for data
            in this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: FullConvGNP model.
    """
    # Make sure that `dim_yc` is initialised and a tuple.
    dim_yc = convert(dim_yc or dim_y, tuple)
    # Make sure that `dim_yt` is initialised.
    dim_yt = dim_yt or dim_y
    # `len(dim_yc)` is equal to the number of density channels. Also add one to account
    # for the identity channel.
    conv_in_channels = sum(dim_yc) + len(dim_yc)

    # This model does not yet support multi-dimensional targets.
    if not (dim_x == 1 and dim_yt == 1):
        raise NotImplementedError(
            "The FullConvGNP for now only supports single-dimensional inputs and "
            "single-dimensional targets."
        )

    # Resolve architecture.
    if conv_arch == "unet":
        pass  # No requirements
    elif conv_arch == "dws":
        if dws_receptive_field is None:
            raise ValueError(f"Must specify `dws_receptive_field`.")
    else:
        raise ValueError(f'Architecture "{conv_arch}" invalid.')

    # If `transform` is set to a value, apply the transform.
    if isinstance(transform, str) and transform.lower() == "positive":
        transform = nps.Transform.positive()
    elif isinstance(transform, tuple):
        lower, upper = transform
        transform = nps.Transform.bounded(lower, upper)
    elif transform is not None:
        raise ValueError(f'Cannot parse value "{transform}" for `transform`.')
    else:
        transform = lambda x: x

    # Construct the core CNN architecture of the model.
    if conv_arch == "unet":
        conv_mean = nps.UNet(
            dim=dim_x,
            in_channels=conv_in_channels,
            out_channels=2,  # Mean and noise
            channels=unet_channels,
            kernels=unet_kernels,
            activations=unet_activations,
            resize_convs=unet_resize_convs,
            resize_conv_interp_method=unet_resize_conv_interp_method,
            dtype=dtype,
        )
        conv_kernel = nps.UNet(
            dim=2 * dim_x,
            in_channels=conv_in_channels + 1,  # Add identity channel.
            out_channels=1,  # Kernel matrix
            channels=unet_channels,
            kernels=unet_kernels,
            activations=unet_activations,
            resize_convs=unet_resize_convs,
            resize_conv_interp_method=unet_resize_conv_interp_method,
            dtype=dtype,
        )
        receptive_field = conv_mean.receptive_field / points_per_unit
    elif conv_arch == "dws":
        conv_mean = nps.ConvNet(
            dim=dim_x,
            in_channels=conv_in_channels,
            out_channels=2,  # Mean and noise
            channels=dws_channels,
            num_layers=dws_layers,
            points_per_unit=points_per_unit,
            receptive_field=dws_receptive_field,
            dtype=dtype,
        )
        conv_kernel = nps.ConvNet(
            dim=2 * dim_x,
            in_channels=conv_in_channels + 1,  # Add identity channel.
            out_channels=1,  # Kernel matrix
            channels=dws_channels,
            num_layers=dws_layers,
            points_per_unit=points_per_unit // 2,  # Keep memory in control.
            receptive_field=dws_receptive_field,
            dtype=dtype,
        )
        receptive_field = dws_receptive_field
    else:
        raise ValueError(f'Architecture "{conv_arch}" invalid.')

    # Construct the discretisation, taking into account that the input to the UNet
    # must play nice with the halving layers.
    disc_mean = nps.Discretisation(
        points_per_unit=points_per_unit,
        multiple=2**conv_mean.num_halving_layers,
        margin=margin,
        dim=dim_x,
    )
    disc_kernel = nps.Discretisation(
        points_per_unit=points_per_unit // 2,  # Keep memory in control.
        multiple=2**conv_kernel.num_halving_layers,
        margin=margin,
        dim=dim_x,  # Only 1D, because the input is later repeated to make it 2D.
    )

    # Construct a separate set conv for every context set.
    encoder_mean_scales = encoder_scales or 2 / disc_mean.points_per_unit
    if not isinstance(encoder_mean_scales, (tuple, list)):
        encoder_mean_scales = (encoder_mean_scales,) * len(dim_yc)
    encoder_mean_set_conv = nps.Parallel(
        *(nps.SetConv(s, dtype=dtype) for s in encoder_mean_scales)
    )
    encoder_kernel_scales = encoder_scales or 2 / disc_kernel.points_per_unit
    if not isinstance(encoder_kernel_scales, (tuple, list)):
        encoder_kernel_scales = (encoder_kernel_scales,) * len(dim_yc)
    # Multiply by two since we halved the PPU.
    encoder_kernel_set_conv = nps.Parallel(
        *(nps.SetConv(2 * s, dtype=dtype) for s in encoder_kernel_scales)
    )

    # Resolve length scales for decoders.
    decoder_mean_scale = decoder_scale or 2 / disc_mean.points_per_unit
    decoder_kernel_scale = decoder_scale or 2 / disc_kernel.points_per_unit
    # Multiply by two since we halved the PPU.
    decoder_kernel_scale *= 2

    # Construct model.
    model = nps.Model(
        nps.Parallel(
            nps.FunctionalCoder(
                disc_mean,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    encoder_mean_set_conv,
                    nps.DivideByFirstChannel(epsilon=epsilon),
                    nps.Materialise(),
                ),
            ),
            nps.FunctionalCoder(
                disc_kernel,
                nps.MapDiagonal(  # Map to diagonal of squared space.
                    nps.Chain(
                        nps.PrependDensityChannel(),
                        encoder_kernel_set_conv,
                        nps.DivideByFirstChannel(epsilon=epsilon),
                        nps.Materialise(),
                        # We only need the identity channel once, so insert it after
                        # materialising.
                        nps.PrependIdentityChannel(),
                    ),
                ),
            ),
        ),
        nps.Chain(
            nps.Parallel(
                nps.Chain(
                    conv_mean,
                    nps.SetConv(decoder_mean_scale, dtype=dtype),
                ),
                nps.MapDiagonal(
                    nps.Chain(
                        conv_kernel,
                        # Ensure that the encoding is PD before smoothing. We cannot
                        # divide by the size of `x`, because that would yield a
                        # changing constant. Instead, we divide by 1000, a constant
                        # meant to stabilise initialisation.
                        lambda x: B.matmul(x, x, tr_b=True) / 1000,
                        nps.SetConv(decoder_kernel_scale, dtype=dtype),
                    ),
                    # The inputs of the encoding already are in the squared space, so
                    # no need to map those again.
                    map_encoding=False,
                ),
            ),
            nps.DenseGaussianLikelihood(),
            transform,
        ),
    )

    # Set attribute `receptive_field`.
    out.kv("Receptive field", receptive_field)
    model.receptive_field = receptive_field

    return model

import wbml.out as out
from plum import convert

import neuralprocesses as nps  # This fixes inspection below.
from .lik import construct_likelihood
from ..util import register_model

__all__ = ["construct_convgnp"]


@register_model
def construct_convgnp(
    dim_x=1,
    dim_xt_aug=None,
    dim_y=1,
    dim_yc=None,
    dim_yt=None,
    points_per_unit=64,
    margin=0.1,
    likelihood="lowrank",
    conv_arch="unet",
    unet_channels=(64,) * 6,
    unet_activations=None,
    unet_kernels=5,
    unet_resize_convs=False,
    unet_resize_conv_interp_method="nearest",
    dws_receptive_field=None,
    dws_layers=8,
    dws_channels=64,
    num_basis_functions=512,
    encoder_scales=None,
    decoder_scale=None,
    xt_aug_layers=(128,) * 3,
    epsilon=1e-4,
    dtype=None,
    nps=nps,
):
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

    # If `dim_xt_aug` is given, contruct an MLP which will use the auxiliary
    # information from the augmented inputs.
    if dim_xt_aug:
        likelihood = nps.Augment(
            nps.Chain(
                nps.MLP(
                    in_dim=conv_out_channels + dim_xt_aug,
                    layers=xt_aug_layers,
                    out_dim=likelihood_in_channels,
                    dtype=dtype,
                ),
                likelihood,
            )
        )
        likelihood_in_channels = unet_channels[0]

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

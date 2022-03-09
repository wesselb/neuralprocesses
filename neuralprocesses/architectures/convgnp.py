from plum import convert
import wbml.out as out

from .lik import construct_likelihood
from ..util import register_model

__all__ = ["construct_convgnp"]


@register_model
def construct_convgnp(nps):
    def construct_convgnp(
        dim_x=1,
        dim_xt_aug=None,
        dim_y=1,
        dim_yc=None,
        dim_yt=None,
        points_per_unit=64,
        margin=0.1,
        likelihood="lowrank",
        unet_channels=(64,) * 6,
        unet_activations=None,
        unet_kernels=5,
        num_basis_functions=512,
        scale=None,
        encoder_scales=None,
        epsilon=1e-4,
        dtype=None,
    ):
        # Make sure that `dim_yc` is initialised and a tuple.
        dim_yc = convert(dim_yc or dim_y, tuple)
        # Make sure that `dim_yt` is initialised.
        dim_yt = dim_yt or dim_y
        # `len(dim_yc)` is equal to the number of density channels.
        unet_in_channels = sum(dim_yc) + len(dim_yc)
        likelihood_in_channels, likelihood = construct_likelihood(
            nps,
            spec=likelihood,
            dim_y=dim_yt,
            num_basis_functions=num_basis_functions,
            dtype=dtype,
        )
        # If `dim_xt_aug` is given, contruct an MLP which will use the auxiliary
        # information from the augmented inputs.
        if dim_xt_aug:
            likelihood = nps.Augment(
                nps.Chain(
                    nps.MLP(
                        dim_in=unet_channels[-1] + dim_xt_aug,
                        dim_hidden=128,
                        dim_out=likelihood_in_channels,
                        num_layers=3,
                        dtype=dtype,
                    ),
                    likelihood,
                )
            )
            likelihood_in_channels = unet_channels[-1]
        # Construct the core CNN architecture of the model.
        unet = nps.UNet(
            dim=dim_x,
            in_channels=unet_in_channels,
            out_channels=likelihood_in_channels,
            channels=unet_channels,
            kernels=unet_kernels,
            activations=unet_activations,
            dtype=dtype,
        )
        # Construct the discretisation, taking into account that the input to the UNet
        # must play nice with the halving layers.
        disc = nps.Discretisation(
            points_per_unit=points_per_unit,
            multiple=2**unet.num_halving_layers,
            margin=margin,
            dim=dim_x,
        )
        out.kv("Receptive field", unet.receptive_field / points_per_unit)
        # Initialise the scale to twice the inter-point spacing for maximum flexibility.
        if scale is None:
            scale = 2 / disc.points_per_unit
        # If `encoder_scales` is not given, use a single set conv. Otherwise, create
        # multiple set convs with their own length scales.
        if encoder_scales is None:
            encoder_set_conv = nps.SetConv(scale, dtype=dtype)
        else:
            if not isinstance(encoder_scales, (tuple, list)):
                encoder_scales = (encoder_scales,) * len(dim_yc)
            encoder_set_conv = nps.Parallel(
                *(nps.SetConv(s, dtype=dtype) for s in encoder_scales)
            )
        return nps.Model(
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
                unet,
                nps.SetConv(scale, dtype=dtype),
                likelihood,
            ),
        )

    return construct_convgnp

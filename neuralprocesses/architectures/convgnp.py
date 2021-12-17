from .lik import construct_likelihood
from ..util import register_model

__all__ = ["construct_convgnp"]


@register_model
def construct_convgnp(nps):
    def construct_convgnp(
        dim_x=1,
        dim_y=1,
        points_per_unit=64,
        margin=0.1,
        likelihood="het",
        num_basis_functions=256,
        harmonics_range=None,
        num_harmonics=64,
        dtype=None,
    ):
        unet_in_channels = dim_y + 1
        unet_out_channels, likelihood = construct_likelihood(
            nps,
            spec=likelihood,
            dim_y=dim_y,
            num_basis_functions=num_basis_functions,
        )
        if harmonics_range is not None:
            harmonics = nps.AppendHarmonics(
                x_range=harmonics_range,
                num_harmonics=num_harmonics,
            )
            unet_in_channels += 2 * num_harmonics
        else:
            harmonics = lambda x: x
        unet = nps.UNet(
            dim=dim_x,
            in_channels=unet_in_channels,
            out_channels=unet_out_channels,
            dtype=dtype,
        )
        disc = nps.Discretisation(
            points_per_unit=points_per_unit,
            multiple=2 ** unet.num_halving_layers,
            margin=margin,
            dim=dim_x,
        )
        return nps.Model(
            nps.FunctionalCoder(
                disc,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.SetConv(disc.points_per_unit, dtype=dtype),
                    nps.DivideByFirstChannel(),
                    harmonics,
                ),
            ),
            nps.Chain(
                unet,
                nps.SetConv(disc.points_per_unit, dtype=dtype),
                likelihood,
            ),
        )

    return construct_convgnp

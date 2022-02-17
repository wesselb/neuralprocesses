from plum import convert

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
        num_basis_functions=512,
        scale=None,
        epsilon=1e-4,
        dtype=None,
    ):
        dim_yc = convert(dim_yc or dim_y, tuple)
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
        unet = nps.UNet(
            dim=dim_x,
            in_channels=unet_in_channels,
            out_channels=likelihood_in_channels,
            channels=unet_channels,
            dtype=dtype,
        )
        disc = nps.Discretisation(
            points_per_unit=points_per_unit,
            multiple=2**unet.num_halving_layers,
            margin=margin,
            dim=dim_x,
        )
        if scale is None:
            scale = 2 / disc.points_per_unit
        return nps.Model(
            nps.FunctionalCoder(
                disc,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.SetConv(scale, dtype=dtype),
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

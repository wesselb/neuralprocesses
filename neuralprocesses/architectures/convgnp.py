import numpy as np

from ..util import register_model

__all__ = ["create_construct_convgnp"]


@register_model
def create_construct_convgnp(ns):
    def construct_convgnp(
        dim_x=1,
        dim_y=1,
        points_per_unit=64,
        margin=0.1,
        likelihood="het",
        num_basis_functions=64,
    ):
        if likelihood == "het":
            unet_out_channels = 2 * dim_y
            likelihood = ns.HeterogeneousGaussianLikelihood()
        elif likelihood == "lowrank":
            unet_out_channels = (2 + num_basis_functions) * dim_y
            likelihood = ns.LowRankGaussianLikelihood(num_basis_functions)
        else:
            raise ValueError(f'Incorrect likelihood "{likelihood}".')
        unet = ns.UNet(
            dim=dim_x,
            in_channels=dim_y + 1,
            out_channels=unet_out_channels,
        )
        disc = ns.Discretisation(
            points_per_unit=points_per_unit,
            multiple=2 ** unet.num_halving_layers,
            margin=margin,
            dim=dim_x,
        )
        return ns.Model(
            ns.FunctionalCoder(
                disc=disc,
                coder=ns.SetConv(disc.points_per_unit, density_channel=True),
            ),
            ns.Chain(
                unet,
                ns.SetConv(disc.points_per_unit),
                likelihood,
            ),
        )

    return construct_convgnp

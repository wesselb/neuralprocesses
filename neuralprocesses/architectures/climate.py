import neuralprocesses as nps  # This fixes inspection below.

from .util import construct_likelihood
from ..util import register_model

__all__ = ["construct_climate_convgnp_mlp", "construct_climate_convgnp_multires"]


@register_model
def construct_climate_convgnp_mlp(
    width_lr=128,
    lr_deg=0.75,
    likelihood="het",
    dtype=None,
    nps=nps,
):
    """Construct a ConvGNP MLP model for climate downscaling.

    References:
        A. Vaughan, W. Tebbutt, J. S. Hosking, R. E. Turner. (2022). ``Convolutional
            Conditional Neural Processes for Local Climate Downscaling,'' in
            Geoscientific Model Development 15(1), pages 251–268. URL:
            https://gmd.copernicus.org/articles/15/251/2022/

    Args:
        width_lr (int, optional): Width of the low-resolution residual network. Defaults
            to 128.
        lr_deg (float, optional): Resolution of the low-resolution grid. Defaults to
            0.75.
        likelihood (str, optional): Likelihood. Must be one of `"het"` or `"lowrank".
            Defaults to `"lowrank"`.
        dtype (dtype, optional): Data type.
    """
    likelihood_in_channels, selector, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=1,
        num_basis_functions=64,
        dtype=dtype,
    )

    conv_lr = nps.ConvNet(
        dim=2,
        in_channels=25,
        channels=width_lr,
        out_channels=128,
        num_layers=6,
        # Force kernel size 3.
        receptive_field=9.5,
        points_per_unit=1 / lr_deg,
        residual=True,
        dtype=dtype,
    )
    assert conv_lr.kernel == 3

    encoder = nps.Chain(
        nps.RestructureParallel(("station", "grid", "elev_grid"), ("grid",)),
        nps.Materialise(),
        nps.DeterministicLikelihood(),
    )

    decoder = nps.Chain(
        conv_lr,
        nps.RepeatForAggregateInputs(
            nps.Chain(
                nps.SetConv(scale=lr_deg, dtype=dtype),
                nps.Augment(
                    nps.MLP(
                        in_dim=128 + 3,
                        layers=(128 + 3, 128, 128),
                        out_dim=likelihood_in_channels,
                        dtype=dtype,
                    ),
                ),
                selector,
            )
        ),
        likelihood,
    )

    return nps.Model(encoder, decoder)


@register_model
def construct_climate_convgnp_multires(
    width_lr=128,
    width_mr=32,
    width_hr=32,
    width_bridge=32,
    lr_deg=0.75,
    mr_deg=0.75 / 7.5,
    hr_deg=0.75 / 75,
    dtype=None,
    likelihood="het",
    nps=nps,
):
    """Construct a multiresolution ConvGNP model for climate downscaling.

    Args:
        width_lr (int, optional): Width of the low-resolution residual network. Defaults
            to 128.
        width_mr (int, optional): Width of the medium-resolution UNet. Defaults to 32.
        width_hr (int, optional): Width of the high-resolution UNet. Defaults to 32.
        width_bridge (int, optional): Number of channels to pass between the
            resolutions. Defaults to 32.
        lr_deg (float, optional): Resolution of the low-resolution grid. Defaults to
            0.75.
        mr_deg (float, optional): Resolution of the medium-resolution grid. Defaults to
            0.1.
        hr_deg (float, optional): Resolution of the high-resolution grid. Defaults to
            0.01.
        likelihood (str, optional): Likelihood. Must be one of `"het"` or `"lowrank".
            Defaults to `"lowrank"`.
        dtype (dtype, optional): Data type.
    """
    likelihood_in_channels, selector, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=1,
        num_basis_functions=64,
        dtype=dtype,
    )

    # Low-resolution CNN:
    conv_lr = nps.ConvNet(
        dim=2,
        # Just the coarse grid:
        in_channels=25 + 1,
        channels=width_lr,
        out_channels=width_bridge,
        num_layers=6,
        # Force kernel size 3.
        receptive_field=9.5,
        points_per_unit=1 / lr_deg,
        residual=True,
        dtype=dtype,
    )
    assert conv_lr.kernel == 3
    disc_lr = nps.Discretisation(
        points_per_unit=1 / lr_deg,
        multiple=1,
        # A margin is not necessary in this case.
        margin=0,
        dim=2,
    )

    # Medium-resolution CNN:
    conv_mr = nps.UNet(
        dim=2,
        # Stations, HR elevation, and low-resolution output:
        in_channels=(1 + 1) + (1 + 1) + width_bridge,
        # Four channels should give a TF of at least ten.
        channels=(width_mr,) * 4,
        out_channels=width_bridge,
        resize_convs=True,
        resize_conv_interp_method="bilinear",
        dtype=dtype,
    )
    assert conv_mr.receptive_field * mr_deg >= 10
    disc_mr = nps.Discretisation(
        points_per_unit=1 / mr_deg,
        multiple=2**conv_mr.num_halving_layers,
        # Use a margin of half the lower bound on the RF.
        margin=5,
        dim=2,
    )

    # High-resolution CNN:
    conv_hr = nps.UNet(
        dim=2,
        # Stations, HR elevation, and medium-resolution output:
        in_channels=(1 + 1) + (1 + 1) + width_bridge,
        # Four channels should give a TF of at least one.
        channels=(width_hr,) * 4,
        out_channels=width_hr,
        resize_convs=True,
        resize_conv_interp_method="bilinear",
        dtype=dtype,
    )
    assert conv_hr.receptive_field * hr_deg >= 1
    disc_hr = nps.Discretisation(
        points_per_unit=1 / hr_deg,
        multiple=2**conv_hr.num_halving_layers,
        # Use a margin of half the lower bound on the RF.
        margin=0.5,
        dim=2,
    )

    encoder = nps.Chain(
        nps.RestructureParallel(
            ("station", "grid", "elev_grid"),
            (
                ("grid",),
                ("station", "elev_grid"),
                ("station", "elev_grid"),
            ),
        ),
        nps.Parallel(
            # Low resolution:
            nps.FunctionalCoder(
                disc_lr,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.Parallel(
                        nps.SetConv(scale=lr_deg, dtype=dtype),
                    ),
                    nps.DivideByFirstChannel(),
                    nps.Materialise(),
                    nps.DeterministicLikelihood(),
                ),
                # Let the discretisation target both the context and target inputs.
                target=lambda xc, xt: (xc, xt),
            ),
            # Medium resolution:
            nps.FunctionalCoder(
                disc_mr,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.Parallel(
                        nps.SetConv(scale=mr_deg, dtype=dtype),
                        nps.SetConv(scale=hr_deg, dtype=dtype),
                    ),
                    nps.DivideByFirstChannel(),
                    nps.Materialise(),
                    nps.DeterministicLikelihood(),
                ),
                # Let the discretisation only target the target inputs.
                target=lambda xc, xt: (xt,),
            ),
            # High resolution:
            nps.FunctionalCoder(
                disc_hr,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.Parallel(
                        nps.SetConv(scale=hr_deg, dtype=dtype),
                        nps.SetConv(scale=hr_deg, dtype=dtype),
                    ),
                    nps.DivideByFirstChannel(),
                    nps.Materialise(),
                    nps.DeterministicLikelihood(),
                ),
                # Let the discretisation only target the target inputs.
                target=lambda xc, xt: (xt,),
            ),
        ),
    )

    decoder = nps.Chain(
        nps.Parallel(
            conv_lr,
            lambda x: x,
            lambda x: x,
        ),
        nps.RestructureParallel(("lr", "mr", "hr"), (("lr", "mr"), "hr")),
        nps.Parallel(
            nps.Chain(
                nps.Fuse(nps.SetConv(scale=lr_deg, dtype=dtype)),
                conv_mr,
            ),
            lambda x: x,
        ),
        nps.Fuse(nps.SetConv(scale=mr_deg, dtype=dtype)),
        conv_hr,
        nps.RepeatForAggregateInputs(
            nps.Chain(
                nps.SetConv(scale=hr_deg, dtype=dtype),
                nps.MLP(
                    in_dim=width_hr,
                    layers=(128, 128, 128),
                    out_dim=likelihood_in_channels,
                ),
                selector,
            )
        ),
        likelihood,
    )

    return nps.Model(encoder, decoder)

import lab as B
import torch

import neuralprocesses.torch as nps
from neuralprocesses.architectures.util import construct_likelihood
from .util import register_data

__all__ = []


def construct_mlp_model(
    width_lr=128,
    lr_deg=0.75,
    likelihood="het",
):
    likelihood_in_channels, selector, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=1,
        num_basis_functions=64,
        dtype=torch.float32,
    )

    conv_lr = nps.ConvNet(
        dim=2,
        in_channels=25,
        channels=width_lr,
        out_channels=128,
        num_layers=6,
        receptive_field=9.5,  # Force kernel size 3.
        points_per_unit=1 / lr_deg,
        residual=True,
    )
    assert conv_lr.kernel == 3

    encoder = nps.Chain(
        nps.RestructureParallel(
            ("station", "grid", "elev_grid"),
            ("grid",),
        ),
        nps.Materialise(),
        nps.DeterministicLikelihood(),
    )
    decoder = nps.Chain(
        conv_lr,
        nps.RepeatForAggregateInputs(
            nps.Chain(
                nps.SetConv(scale=lr_deg),
                nps.Augment(
                    nps.MLP(
                        in_dim=128 + 3,
                        layers=(128 + 3, 128, 128),
                        out_dim=likelihood_in_channels,
                    ),
                ),
                selector,
            )
        ),
        likelihood,
    )

    return nps.Model(encoder, decoder)


class Fuse(torch.nn.Module):
    def __init__(self, set_conv):
        super().__init__()
        self.set_conv = set_conv


@nps.code.dispatch
def code(coder: Fuse, xz, z, x, **kw_args):
    xz1, xz2 = xz
    z1, z2 = z
    _, z2 = nps.code(coder.set_conv, xz2, z2, xz1, **kw_args)
    return xz1, B.concat(z1, z2, axis=-1 - nps.data_dims(xz1))


def construct_multires_model(
    width_hr=64,
    width_mr=64,
    width_lr=128,
    width_bridge=32,
    hr_deg=0.75 / 75,
    mr_deg=0.75 / 7.5,
    lr_deg=0.75,
    likelihood="het",
):
    likelihood_in_channels, selector, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=1,
        num_basis_functions=64,
        dtype=torch.float32,
    )

    # High-resolution CNN:
    conv_hr = nps.UNet(
        dim=2,
        in_channels=(1 + 1) + (1 + 1) + width_bridge,  # Stations and HR elevation
        # Four channels should give a TF of at least one.
        channels=(width_hr,) * 4,
        out_channels=likelihood_in_channels,
    )
    assert conv_hr.receptive_field * hr_deg >= 1
    disc_hr = nps.Discretisation(
        points_per_unit=1 / hr_deg,
        multiple=2**conv_hr.num_halving_layers,
        # Use a margin of half the lower bound on the RF.
        margin=0.5,
        dim=2,
    )

    # Medium-resolution CNN:
    conv_mr = nps.UNet(
        dim=2,
        in_channels=(1 + 1) + (1 + 1) + width_bridge,  # Stations and HR elevation
        # Four channels should give a TF of at least ten.
        channels=(width_mr,) * 4,
        out_channels=width_bridge,
    )
    assert conv_mr.receptive_field * mr_deg >= 10
    disc_mr = nps.Discretisation(
        points_per_unit=1 / mr_deg,
        multiple=2**conv_mr.num_halving_layers,
        # Use a margin of half the lower bound on the RF.
        margin=5,
        dim=2,
    )

    # Low-resolution CNN:
    conv_lr = nps.ConvNet(
        dim=2,
        in_channels=(25 + 1) + (1 + 1),  # Coarse grid and HR elevation
        channels=width_lr,
        out_channels=width_bridge,
        num_layers=6,
        receptive_field=9.5,  # Force kernel size 3.
        points_per_unit=1 / lr_deg,
        residual=True,
    )
    assert conv_lr.kernel == 3
    disc_lr = nps.Discretisation(
        points_per_unit=1 / lr_deg,
        multiple=1,
        # A margin is not necessary in this case.
        margin=0,
        dim=2,
    )

    encoder = nps.Chain(
        nps.RestructureParallel(
            ("station", "grid", "elev_grid"),
            (
                ("station", "elev_grid"),
                ("station", "elev_grid"),
                ("grid", "elev_grid"),
            ),
        ),
        nps.Parallel(
            # High resolution:
            nps.FunctionalCoder(
                disc_hr,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.Parallel(
                        nps.SetConv(scale=hr_deg),
                        nps.SetConv(scale=hr_deg),
                    ),
                    nps.DivideByFirstChannel(),
                    nps.Materialise(),
                    nps.DeterministicLikelihood(),
                ),
                # Let the discretisation only target the target inputs.
                target=lambda xc, xt: (xt,),
            ),
            # Medium resolution:
            nps.FunctionalCoder(
                disc_mr,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.Parallel(
                        nps.SetConv(scale=mr_deg),
                        nps.SetConv(scale=hr_deg),
                    ),
                    nps.DivideByFirstChannel(),
                    nps.Materialise(),
                    nps.DeterministicLikelihood(),
                ),
                # Let the discretisation only target the target inputs.
                target=lambda xc, xt: (xt,),
            ),
            # Low resolution:
            nps.FunctionalCoder(
                disc_lr,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.Parallel(
                        nps.SetConv(scale=lr_deg),
                        nps.SetConv(scale=hr_deg),
                    ),
                    nps.DivideByFirstChannel(),
                    nps.Materialise(),
                    nps.DeterministicLikelihood(),
                ),
                # Let the discretisation target both the context and target inputs.
                target=lambda xc, xt: (xc, xt),
            ),
        ),
    )

    decoder = nps.Chain(
        nps.Parallel(
            lambda x: x,
            lambda x: x,
            conv_lr,
        ),
        nps.RestructureParallel((0, 1, 2), ((0,), (1, 2))),
        nps.Parallel(
            lambda x: x,
            Fuse(nps.SetConv(scale=lr_deg)),
        ),
        nps.RestructureParallel(((0,), 1), (0, 1)),
        nps.Parallel(
            lambda x: x,
            conv_mr,
        ),
        Fuse(nps.SetConv(scale=mr_deg)),
        conv_hr,
        nps.RepeatForAggregateInputs(
            nps.Chain(
                nps.SetConv(scale=hr_deg),
                selector,
            )
        ),
        likelihood,
    )

    return nps.Model(encoder, decoder)


def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    config["dim_x"] = 2
    config["dim_y"] = 1

    if args.model == "convcnp-mlp":
        config["model"] = construct_mlp_model(likelihood="het")
        target_elev = True
        do_plot = False
    elif args.model == "convgnp-mlp":
        config["model"] = construct_mlp_model(likelihood="lowrank")
        target_elev = True
        do_plot = False
    elif args.model == "convcnp-multires":
        config["model"] = construct_multires_model(likelihood="het")
        target_elev = False
        do_plot = True
    elif args.model == "convgnp-multires":
        config["model"] = construct_multires_model(likelihood="lowrank")
        target_elev = False
        do_plot = True
    else:
        raise ValueError(f'Experiment does not yet support model "{args.model}".')

    # Other settings specific to the temperature experiment:
    if do_plot:
        config["plot"] = {2: {"range": ((6, 16), (47, 55))}}
    else:
        config["plot"] = {}

    gen_train = nps.TemperatureGenerator(
        torch.float32,
        seed=10,
        batch_size=args.batch_size,
        context_fraction=0.5,
        target_min=5,
        target_square=2,
        target_elev=target_elev,
        subset="train",
        device=device,
    )
    gen_cv = lambda: nps.TemperatureGenerator(
        torch.float32,
        seed=20,
        batch_size=args.batch_size,
        context_fraction=0.5,
        target_min=5,
        target_square=2,
        target_elev=target_elev,
        subset="cv",
        # Cycle over the data a few times to account for the random square sampling.
        passes=5,
        device=device,
    )
    gens_eval = lambda: [
        (
            "Evaluation",
            nps.TemperatureGenerator(
                torch.float32,
                seed=30,
                batch_size=args.batch_size,
                context_fraction=0,  # Don't sample contexts.
                target_min=1,
                # Don't sample squares, but use the whole data.
                target_square=2 if "eval-square" in args.experiment_setting else 0,
                target_elev=target_elev,
                subset="eval",
                device=device,
            ),
        )
    ]
    return gen_train, gen_cv, gens_eval


register_data("temperature", setup)

import lab as B
from stheno import EQ, Matern52

from .gp import GPGenerator
from .mixture import MixtureGenerator
from .mixgp import MixtureGPGenerator
from .sawtooth import SawtoothGenerator
from ..dist.uniform import UniformDiscrete, UniformContinuous

__all__ = ["construct_predefined_gens"]


def construct_predefined_gens(
    dtype,
    seed=0,
    batch_size=16,
    num_tasks=2**14,
    dim_x=1,
    dim_y=1,
    x_range_context=(-2, 2),
    x_range_target=(-2, 2),
    mean_diff=0.0,
    pred_logpdf=True,
    pred_logpdf_diag=True,
    device="cpu",
):
    """Construct a number of predefined data generators.

    Args:
        dtype (dtype): Data type to generate.
        seed (int, optional): Seed. Defaults to `0`.
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an
            integer multiple of `batch_size`. Defaults to 2^14.
        dim_x (int, optional): Dimensionality of the input space. Defaults to `1`.
        dim_y (int, optional): Dimensionality of the output space. Defaults to `1`.
        x_range_context (tuple[float, float], optional): Range of the inputs of the
            context points. Defaults to `(-2, 2)`.
        x_range_target (tuple[float, float], optional): Range of the inputs of the
            target points. Defaults to `(-2, 2)`.
        mean_diff (float, optional): Difference in means in the samples of
            :class:`neuralprocesses.data.mixgp.MixtureGPGenerator`.
        pred_logpdf (bool, optional): Also compute the logpdf of the target set given
            the context set under the true GP. Defaults to `True`.
        pred_logpdf_diag (bool, optional): Also compute the logpdf of the target set
            given the context set under the true diagonalised GP. Defaults to `True`.
        device (str, optional): Device on which to generate data. Defaults to `cpu`.

    Returns:
        dict: A dictionary mapping names of data generators to the generators.
    """
    # Ensure that distances don't become bigger as we increase the input dimensionality.
    # We achieve this by blowing up all length scales by `sqrt(dim_x)`.
    factor = B.sqrt(dim_x)
    config = {
        "num_tasks": num_tasks,
        "batch_size": batch_size,
        "dist_x_context": UniformContinuous(*((x_range_context,) * dim_x)),
        "dist_x_target": UniformContinuous(*((x_range_target,) * dim_x)),
        "dim_y": dim_y,
        "device": device,
    }
    kernels = {
        "eq": EQ().stretch(factor * 0.25),
        "matern": Matern52().stretch(factor * 0.25),
        "weakly-periodic": (
            EQ().stretch(factor * 0.5) * EQ().stretch(factor).periodic(factor * 0.25)
        ),
    }
    gens = {
        name: GPGenerator(
            dtype,
            seed=seed,
            noise=0.05,
            kernel=kernel,
            num_context=UniformDiscrete(0, 30 * dim_x),
            num_target=UniformDiscrete(50 * dim_x, 50 * dim_x),
            pred_logpdf=pred_logpdf,
            pred_logpdf_diag=pred_logpdf_diag,
            **config,
        )
        for name, kernel in kernels.items()
    }
    # Previously, the maximum number of context points was `75 * dim_x`. However, if
    # `dim_x == 1`, then this is too high. We therefore change that case, and keep all
    # other cases the same.
    max_context = 30 if dim_x == 1 else 75 * dim_x
    gens["sawtooth"] = SawtoothGenerator(
        dtype,
        seed=seed,
        # The sawtooth is hard already as it is. Do not add noise.
        noise=0,
        dist_freq=UniformContinuous(2 / factor, 4 / factor),
        num_context=UniformDiscrete(0, max_context),
        num_target=UniformDiscrete(100 * dim_x, 100 * dim_x),
        **config,
    )
    # Be sure to use different seeds in the mixture components.
    gens["mixture"] = MixtureGenerator(
        *(
            GPGenerator(
                dtype,
                seed=seed + i,
                noise=0.05,
                kernel=kernel,
                num_context=UniformDiscrete(0, max_context),
                num_target=UniformDiscrete(100 * dim_x, 100 * dim_x),
                pred_logpdf=pred_logpdf,
                pred_logpdf_diag=pred_logpdf_diag,
                **config,
            )
            # Make sure that the order of `kernels.items()` is fixed.
            for i, (_, kernel) in enumerate(sorted(kernels.items(), key=lambda x: x[0]))
        ),
        SawtoothGenerator(
            dtype,
            seed=seed + len(kernels.items()),
            # The sawtooth is hard already as it is. Do not add noise.
            noise=0,
            dist_freq=UniformContinuous(2 / factor, 4 / factor),
            num_context=UniformDiscrete(0, max_context),
            num_target=UniformDiscrete(100 * dim_x, 100 * dim_x),
            **config,
        ),
        seed=seed,
    )

    for i, kernel in enumerate(kernels.keys()):
        gens[f"mix-{kernel}"] = MixtureGPGenerator(
            dtype,
            seed=seed + len(kernels.items()) + i + 1,
            noise=0.05,
            kernel=kernels[kernel],
            num_context=UniformDiscrete(0, 30 * dim_x),
            num_target=UniformDiscrete(50 * dim_x, 50 * dim_x),
            pred_logpdf=False,
            pred_logpdf_diag=False,
            mean_diff=mean_diff,
            **config,
        )

    return gens

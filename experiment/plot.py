import lab as B
import matplotlib.pyplot as plt
import neuralprocesses.torch as nps
import stheno
import torch
from wbml.plot import tweak

__all__ = ["visualise"]


def visualise(model, gen, *, path, config, predict=nps.predict):
    """Plot the prediction for the first element of a batch."""
    if config["dim_x"] == 1:
        visualise_1d(
            model,
            gen,
            path=path,
            config=config,
            predict=predict,
        )
    elif config["dim_x"] == 2:
        visualise_2d(
            model,
            gen,
            path=path,
            config=config,
            predict=predict,
        )
    else:
        pass  # Not implemented. Just do nothing.


def visualise_1d(model, gen, *, path, config, predict):
    batch = nps.batch_index(gen.generate_batch(), slice(0, 1, None))

    try:
        plot_config = config["plot"][1]
    except KeyError:
        return

    # Define points to predict at.
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), *plot_config["range"], 200)

    # Predict with model.
    with torch.no_grad():
        mean, var, samples, _ = predict(
            model,
            batch["contexts"],
            nps.AggregateInput(
                *((x[None, None, :], i) for i in range(config["dim_y"]))
            ),
        )

    plt.figure(figsize=(8, 6 * config["dim_y"]))

    for i in range(config["dim_y"]):
        plt.subplot(config["dim_y"], 1, 1 + i)

        # Plot context and target.
        plt.scatter(
            nps.batch_xc(batch, i)[0, 0],
            nps.batch_yc(batch, i)[0],
            label="Context",
            style="train",
            s=20,
        )

        plt.scatter(
            nps.batch_xt(batch, i)[0, 0],
            nps.batch_yt(batch, i)[0],
            label="Target",
            style="test",
            s=20,
        )

        # Plot prediction.
        err = 1.96 * B.sqrt(var[i][0, 0])
        plt.plot(
            x,
            mean[i][0, 0],
            label="Prediction",
            style="pred",
        )
        plt.fill_between(
            x,
            mean[i][0, 0] - err,
            mean[i][0, 0] + err,
            style="pred",
        )
        plt.plot(
            x,
            B.transpose(samples[i][:10, 0, 0]),
            style="pred",
            ls="-",
            lw=0.5,
        )

        # Plot prediction by ground truth.
        if hasattr(gen, "kernel") and config["dim_y"] == 1:
            f = stheno.GP(gen.kernel)
            # Make sure that everything is of `float64`s and on the GPU.
            noise = B.to_active_device(B.cast(torch.float64, gen.noise))
            xc = B.cast(torch.float64, nps.batch_xc(batch, 0)[0, 0])
            yc = B.cast(torch.float64, nps.batch_yc(batch, 0)[0])
            x = B.cast(torch.float64, x)
            # Compute posterior GP.
            f_post = f | (f(xc, noise), yc)
            mean, lower, upper = f_post(x).marginal_credible_bounds()
            plt.plot(x, mean, label="Truth", style="pred2")
            plt.plot(x, lower, style="pred2")
            plt.plot(x, upper, style="pred2")

        for x_axvline in plot_config["axvline"]:
            plt.axvline(x_axvline, c="k", ls="--", lw=0.5)

        plt.xlim(B.min(x), B.max(x))
        tweak()

    plt.savefig(path)
    plt.close()


def visualise_2d(model, gen, *, path, config, predict):
    batch = nps.batch_index(gen.generate_batch(), slice(0, 1, None))

    try:
        plot_config = config["plot"][2]
    except KeyError:
        return

    # Define points to predict at.
    with B.on_device(batch["xt"]):
        # For now we use list form, because not all components yet support tuple
        # specification.
        # TODO: Use tuple specification when it is supported everywhere.
        n = 100
        x0 = B.linspace(B.dtype(batch["xt"]), *plot_config["range"][0], n)
        x1 = B.linspace(B.dtype(batch["xt"]), *plot_config["range"][1], n)
        x0 = B.flatten(B.broadcast_to(x0[:, None], n, n))
        x1 = B.flatten(B.broadcast_to(x1[None, :], n, n))
        x_list = B.stack(x0, x1)

    # Predict with model and produce five noiseless samples.
    with torch.no_grad():
        mean, _, samples, _ = predict(
            model,
            batch["contexts"],
            nps.AggregateInput(
                *((x_list[None, :, :], i) for i in range(config["dim_y"]))
            ),
            num_samples=2,
        )

    vmin = max(B.max(mean), B.max(samples))
    vmax = min(B.min(mean), B.min(samples))

    def plot_imshow(image, i, label):
        image = B.reshape(image, n, n)
        plt.imshow(
            B.transpose(image),
            cmap="viridis",
            vmin=vmax,
            vmax=vmin,
            origin="lower",
            extent=plot_config["range"][0] + plot_config["range"][1],
            label=label,
        )
        plt.scatter(
            nps.batch_xc(batch, i)[0, 0],
            nps.batch_xc(batch, i)[0, 1],
            c=nps.batch_yc(batch, i)[0],
            cmap="viridis",
            vmin=vmax,
            vmax=vmin,
            edgecolor="white",
            linewidth=0.5,
            s=80,
            label="Context",
        )
        plt.scatter(
            nps.batch_xt(batch, i)[0, 0],
            nps.batch_xt(batch, i)[0, 1],
            c=nps.batch_yt(batch, i)[0],
            cmap="viridis",
            vmin=vmax,
            vmax=vmin,
            edgecolor="black",
            linewidth=0.5,
            s=80,
            marker="d",
            label="Target",
        )
        # Remove ticks, because those are noisy.
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

    plt.figure(figsize=(12, 4 * config["dim_y"]))

    for i in range(config["dim_y"]):
        # Plot mean.
        plt.subplot(config["dim_y"], 3, 1 + i * 3)
        if i == 0:
            plt.title("Mean")
        plot_imshow(mean[i][0, 0], i, label="Mean")
        tweak(grid=False)

        # Plot first sample.
        plt.subplot(config["dim_y"], 3, 2 + i * 3)
        if i == 0:
            plt.title("Sample 1")
        plot_imshow(samples[i][0, 0, 0], i, label="Sample")
        tweak(grid=False)

        # Plot second sample.
        plt.subplot(config["dim_y"], 3, 3 + i * 3)
        if i == 0:
            plt.title("Sample 2")
        plot_imshow(samples[i][1, 0, 0], i, label="Sample")
        tweak(grid=False)

    plt.savefig(path)
    plt.close()

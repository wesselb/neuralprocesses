import argparse
import asyncio
import asyncio.subprocess
import os
import signal
import subprocess

import wbml.out as out

spawned = []


def read_values(xs, sep, *values):
    """Read values from a string.

    Args:
        xs (str): Values as one string.
        sep (str): Separator separating the values in the string.
        *values (str): Names of the values.

    Returns:
        dict: Naming of the values mapping to the values.
    """
    xs = [x.strip() for x in xs.split(sep)]
    if len(xs) != len(values):
        raise ValueError(f"Expected {len(values)} values, but got {len(xs)}.")
    return {v: int(x) for v, x in zip(values, xs)}


def nvidia_smi(gpu_id):
    """Run `nvidia-smi`.

    Args:
        gpu_id (int): GPU ID.

    Returns:
        dict: Statistics of GPU `gpu_id`.
    """
    p = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
    )
    res, _ = p.communicate()
    res = res.decode().strip()
    stats = [read_values(x, ",", "utilisation", "memory") for x in res.splitlines()]
    return stats[gpu_id]


def dict_subtract(d1, d2):
    """Subtract one dictionary from another.

    Args:
        d1 (dict): First dictionary.
        d2 (dict): Second dictionary.

    Returns:
        dict: `d1 - d2`.
    """
    if set(d1.keys()) != set(d2.keys()):
        raise ValueError("Dictionaries have different keys.")
    return {k: d1[k] - d2[k] for k in d1.keys()}


def dict_max(*ds):
    """Take the maximum of dictionaries.

    Args:
        *ds (dict): Dictionaries.

    Returns:
        dict: `max(*ds)`.
    """
    if not all([set(d.keys()) == set(ds[0].keys()) for d in ds[1:]]):
        raise ValueError("Dictionaries have different keys.")
    return {k: max([d[k] for d in ds]) for k in ds[0].keys()}


async def benchmark_command(gpu_id, command):
    """Benchmark a command on the GPU.

    Args:
        gpu_id (int): GPU to run the command on.
        command (str): Command to benchmark.

    Returns:
        dict: Statistics of `command` on GPU `gpu_id`.
    """
    with out.Section("Benchmarking command"):
        out.kv("Command", command)

        # Start process.
        stats_before = nvidia_smi(gpu_id)
        p = await asyncio.create_subprocess_shell(
            f"CUDA_VISIBLE_DEVICES={gpu_id} " + command,
            preexec_fn=os.setsid,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        spawned.append(p)

        stats_diff = dict_subtract(await determine_current_stats(gpu_id), stats_before)

        # Kill the process.
        if p.returncode is None:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            out.out(f"Killed PID {p.pid}.")
        else:
            raise RuntimeError("Process already terminated. Something went wrong!")
        # Wait five seconds for the process to shut down.
        await asyncio.sleep(5)

        return stats_diff


async def determine_current_stats(gpu_id):
    """Determine the current statistics of GPU `gpu_id` by monitoring the GPU over 20
    seconds.

    Args:
        gpu_id (int): GPU ID.

    Returns:
        dict: Statistics of GPU `gpu_id`.
    """
    stats = []
    current = 0
    while current < 20:
        await asyncio.sleep(1)
        stats.append(nvidia_smi(gpu_id))
        current += 1
    return dict_max(*stats)


def test_success(command):
    """Test whether a command exists successfully.

    Args:
        command (str): Command.

    Returns:
        bool: Success of command `command`.
    """
    try:
        subprocess.check_output(command, shell=True, stderr=asyncio.subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


async def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--data",
        choices=[
            "eq",
            "matern",
            "weakly-periodic",
            "sawtooth",
            "mixture",
        ],
        required=True,
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--memory", type=int, default=11_019)
    args = parser.parse_args()

    def with_gpu(c):
        return f"CUDA_VISIBLE_DEVICES={args.gpu} {c}"

    # Setup script.
    out.report_time = True

    # Determine the suite of experiments to run.
    commands = (
        # Conditional models:
        [
            f"python train_gp.py"
            f" --model {model}"
            f" --data {args.data}"
            f" --dim-x {dim_x}"
            f" --dim-y {dim_y}"
            f" --epochs 100"
            f" --batch-size 16"
            f" --rate 3e-4"
            for dim_x in [1, 2]
            for dim_y in [1, 2]
            for model in ["cnp", "acnp", "convcnp", "gnp", "agnp", "convgnp"]
            if not (model == "convgnp" and dim_x == dim_y == 2)
        ]
        # The ConvGNP for 2D inputs and 2D outputs just doesn't fit in memory. Reduce
        # the batch size and learning rate by a factor two.
        + [
            f"python train_gp.py"
            f" --model convgnp"
            f" --data {args.data}"
            f" --dim-x 2"
            f" --dim-y 2"
            f" --epochs 100"
            f" --batch-size 8"
            f" --rate 1.5e-4"
        ]
        # FullConvGNP:
        + [
            f"python train_gp.py"
            f" --model fullconvgnp"
            f" --data {args.data}"
            f" --dim-x 1"
            f" --dim-y 1"
            f" --epochs 100"
            f" --batch-size 16"
            f" --rate 3e-4"
        ]
        # Latent-variable models:
        + [
            f"python train_gp.py"
            f" --model {model}"
            f" --objective {objective}"
            f" --data {args.data}"
            f" --dim-x {dim_x}"
            f" --dim-y {dim_y}"
            f" --epochs 100"
            f" --batch-size 16"
            f" --rate 3e-4"
            for dim_x in [1, 2]
            for dim_y in [1, 2]
            for model in ["np", "anp", "convnp"]
            if not (model == "convnp" and dim_x == 2)
            for objective in [
                f"loglik --num-samples 20",
                f"elbo --num-samples 5",
            ]
        ]
        # The ConvNP for 2D inputs is too expensive and doesn't fit in memory. We reduce
        # the numbers of samples to keep the memory and runtime in check.
        + [
            f"python train_gp.py"
            f" --model convnp"
            f" --objective {objective}"
            f" --data {args.data}"
            f" --dim-x 2"
            f" --dim-y {dim_y}"
            f" --epochs 100"
            f" --batch-size 16"
            f" --rate 3e-4"
            for dim_y in [1, 2]
            for objective in [
                f"loglik --num-samples 5",
                f"elbo --num-samples 1",
            ]
        ]
    )
    if args.evaluate:
        commands = [c + " --evaluate" for c in commands]

    if not args.evaluate:
        # Run through the commands and eject the ones that have already completed.
        for c in list(commands):  # Copy, because we're removing as we go!
            if test_success(with_gpu(c + " --check-completed")):
                with out.Section("Command already completed"):
                    out.kv("Command", c)
                commands.remove(c)

    # Benchmark every command before commit to the long run.
    benchmark = {c: await benchmark_command(args.gpu, c) for c in commands}

    # Sort the commands by utilisation then memory.
    commands = sorted(
        commands,
        key=lambda c: (benchmark[c]["utilisation"], benchmark[c]["memory"]),
    )
    with out.Section("Commands"):
        for c in commands:
            out.out(c)

    while commands:
        # Check which commands we can run without putting too much strain on the GPU.
        stats = await determine_current_stats(args.gpu)

        eligible_commands = []
        for c in commands:
            # Predict 10% more memory usage than the benchmark. Also leave 10% room
            # to be sure.
            if stats["memory"] + 1.10 * benchmark[c]["memory"] > 0.9 * args.memory:
                # Takes too much memory.
                continue
            if stats["utilisation"] + benchmark[c]["utilisation"] > 120:
                # Fine to max out the GPU, but not much more than that.
                continue
            eligible_commands.append(c)

        if eligible_commands:
            # Decide on the first eligible command.
            c = eligible_commands[0]
            with out.Section("Running command"):
                out.kv("Command", c)
            p = await asyncio.create_subprocess_shell(
                with_gpu(c),
                preexec_fn=os.setsid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            commands.remove(c)
            spawned.append(p)
            out.kv("Remaining", len(commands))

    # Wait for all spawned processes to finish before exiting the script.
    out.out("Waiting for processes to finish...")
    for p in spawned:
        if p.returncode is None:
            await p.wait()
    out.out("Done!")


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        # If the user kills the program, kill all spawned processes.
        with out.Section("Killing all spawned processes"):
            for p in spawned:
                if p.returncode is None:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    out.out(f"Killed PID {p.pid}.")

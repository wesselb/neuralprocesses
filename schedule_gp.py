import argparse
import asyncio
import asyncio.subprocess
import os
import signal
import subprocess

import wbml.out as out

spawned = []


def read_values(xs, sep, *values):
    xs = [x.strip() for x in xs.split(sep)]
    if len(xs) != len(values):
        raise ValueError(f"Expected {len(values)} values, but got {len(xs)}.")
    return {v: int(x) for v, x in zip(values, xs)}


def nvidia_smi(gpu_id):
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


def dict_diff(d1, d2):
    if set(d1.keys()) != set(d2.keys()):
        raise ValueError("Dictionaries have different keys.")
    return {k: d1[k] - d2[k] for k in d1.keys()}


def dict_max(*ds):
    if not all([set(d.keys()) == set(ds[0].keys()) for d in ds[1:]]):
        raise ValueError("Dictionaries have different keys.")
    return {k: max([d[k] for d in ds]) for k in ds[0].keys()}


async def benchmark_command(gpu_id, command):
    with out.Section("Benchmarking command"):
        out.kv("Command", command)

        # Start process.
        stats_before = nvidia_smi(gpu_id)
        p = await asyncio.create_subprocess_shell(
            f"CUDA_VISIBLE_DEVICES={gpu_id} " + command,
            preexec_fn=os.setsid,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        spawned.append(p)

        stats_diff = dict_diff(await determine_current_stats(gpu_id), stats_before)

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
    stats = []
    current = 0
    while current < 20:
        await asyncio.sleep(1)
        stats.append(nvidia_smi(gpu_id))
        current += 1
    return dict_max(*stats)


async def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--data",
        choices=["eq", "matern", "weakly-periodic", "sawtooth", "mixture"],
        required=True,
    )
    parser.add_argument("--memory", type=int, default=11_019)
    args = parser.parse_args()

    # Setup script.
    out.report_time = True

    # Determine the suite of experiments to run.
    conditional_commands = [
        f"python train_gp.py"
        f" --model {model}"
        f" --data {args.data}"
        f" --dim-x {dim_x}"
        f" --dim-y {dim_y}"
        f" --epochs 100"
        for dim_x in [1, 2]
        for dim_y in [1, 2]
        for model in ["cnp", "acnp", "convcnp", "gnp", "agnp", "convgnp"]
    ] + [
        f"python train_gp.py"
        f" --model fullconvgnp"
        f" --data {args.data}"
        f" --dim-x 1"
        f" --dim-y 1"
        f" --epochs 100"
    ]
    lv_commands = [
        f"python train_gp.py"
        f" --model {model}"
        f" --objective {objective}"
        f" --data {args.data}"
        f" --dim-x {dim_x}"
        f" --dim-y {dim_y}"
        f" --epochs 100"
        for dim_x in [1, 2]
        for dim_y in [1, 2]
        for model in ["np", "anp", "convnp"]
        for objective in ["loglik --num-samples 20", "elbo --num-samples 5"]
    ]

    # Benchmark every command before commit to the long run.
    benchmark = {
        c: await benchmark_command(args.gpu, c)
        for c in conditional_commands + lv_commands
    }

    # Sort the commands by memory then utilisation. Run the conditional models before
    # running the latent-variable models.
    conditional_commands = sorted(
        conditional_commands,
        key=lambda c: (benchmark[c]["memory"], benchmark[c]["utilisation"]),
    )
    lv_commands = sorted(
        lv_commands,
        key=lambda c: (benchmark[c]["memory"], benchmark[c]["utilisation"]),
    )
    commands = conditional_commands + lv_commands
    with out.Section("Commands"):
        for c in commands:
            out.out(c)

    while commands:
        # Check which commands we can run without putting too much strain on the GPU.
        stats = await determine_current_stats(args.gpu)

        eligible_commands = []
        for c in commands:
            # Predict 15% more memory usage than the benchmark. Also leave 10% room
            # to be sure.
            if stats["memory"] + 1.15 * benchmark[c]["memory"] > 0.9 * args.memory:
                # Takes too much memory.
                continue
            if stats["utilisation"] >= 95:
                # GPU already basically maxed out.
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
                f"CUDA_VISIBLE_DEVICES={args.gpu} " + c,
                preexec_fn=os.setsid,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            commands.remove(c)
            spawned.append(p)
            out.kv("Remaining", len(commands))


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

    except KeyboardInterrupt:
        out.out("Killing all spawned processes.")
        for p in spawned:
            if p.returncode is None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                out.out(f"Killed PID {p.pid}.")

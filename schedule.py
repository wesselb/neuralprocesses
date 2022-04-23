import asyncio
import asyncio.subprocess
import os
import signal
import subprocess
import argparse

import wbml.out as out
from wbml.experiment import WorkingDirectory


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


async def benchmark(gpu_id, command):
    with out.Section("Determining GPU usage"):
        out.kv("Command", command)

        out.out("Starting process.")
        stats_before = nvidia_smi(gpu_id)
        p = await asyncio.create_subprocess_shell(
            command,
            preexec_fn=os.setsid,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        sleep_total = 10
        out.out(f"Monitoring usage over {sleep_total} seconds.")
        sleep_current = 0
        stats_collected = []
        while sleep_current < sleep_total:
            await asyncio.sleep(1)
            stats_collected.append(nvidia_smi(gpu_id))
            sleep_current += 1

        if p.returncode is None:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            out.out("Process killed.")
        else:
            raise RuntimeError("Process already terminated. Something went wrong!")

        stats_diff = dict_max(*(dict_diff(d, stats_before) for d in stats_collected))
        out.kv("Collected statistics", stats_diff)

        return stats_diff


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--benchmark", action="action_store")
    args = parser.parse_args()

    wd = WorkingDirectory("_scheduling")

    commands = [
        f"CUDA_VISIBLE_DEVICES={args.gpu} python train_gp.py --model convcnp --data eq",
        f"CUDA_VISIBLE_DEVICES={args.gpu} python train_gp.py --model convcnp --data matern",
    ]

    if args.benchmark:
        benchmark = {}
        for command in commands:
            benchmark[command] = await benchmark(args.gpu, command)
        wd.save(benchmark, "benchmark.pickle")

    benchmark = wd.load("benchmark.pickle")
    print(benchmark)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

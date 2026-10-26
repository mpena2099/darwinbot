"""
tools/kaggle/bench_gpu.py
-------------------------
Episodes per second on the CPU (MuJoCo, every core) against the GPU (MJX,
batched), for one body and growing batch sizes. Answers whether a GPU speeds
up training, and from which CMA-ES population size.

  python tools/kaggle/bench_gpu.py                       # CPU and, if jax sees one, GPU
  python tools/kaggle/bench_gpu.py --sizes 16,64,256 --terrain rough

The GPU compiles a program for every new body; that time is reported apart,
because training pays it once per body.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np   # noqa: E402

from darwinbot import morphology as mo   # noqa: E402
from darwinbot.controller import random_controller   # noqa: E402
from darwinbot.sim import Episode   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402

BODIES = {"quadruped": lambda: mo.template(2),
          "hexapod": lambda: mo.template(3),
          "wheeled": lambda: mo.template(2, wheels=True)}


def _cpu_chunk(job):
    body, terrain, seconds, ctrls = job
    ep = Episode(BODIES[body](), get_terrains(terrain)[0], seconds)
    for c in ctrls:
        ep.run(c)
    return len(ctrls)


def bench_cpu(body, terrain, seconds, n, workers):
    rng = np.random.default_rng(0)
    ctrls = [random_controller(rng) for _ in range(n)]
    chunks = [(body, terrain, seconds, ctrls[i::workers]) for i in range(workers)]
    t = time.perf_counter()
    with Pool(workers) as pool:
        done = sum(pool.map(_cpu_chunk, chunks))
    return done / (time.perf_counter() - t)


def bench_gpu(body, terrain, seconds, sizes):
    from darwinbot.sim_gpu import BatchEpisode
    rng = np.random.default_rng(0)
    ep = BatchEpisode(BODIES[body](), get_terrains(terrain)[0], seconds)
    rows = []
    for k, n in enumerate(sizes):
        ctrls = np.array([random_controller(rng) for _ in range(n)])
        t = time.perf_counter()
        ep.run_batch(ctrls)                   # first call of a size compiles
        first = time.perf_counter() - t
        t = time.perf_counter()
        ep.run_batch(ctrls)
        rate = n / (time.perf_counter() - t)
        rows.append((n, first - n / rate, rate))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--body", choices=BODIES, default="quadruped")
    p.add_argument("--terrain", default="flat")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--sizes", default="16,64,256,1024")
    p.add_argument("--cpu-episodes", type=int, default=48)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]

    print(f"body {a.body}, terrain {a.terrain}, {a.seconds:g} s episodes")
    cpu = bench_cpu(a.body, a.terrain, a.seconds, a.cpu_episodes, a.workers)
    print(f"CPU, {a.workers} cores: {cpu:7.1f} episodes/s")

    try:
        import jax
        devices = jax.devices()
    except Exception as err:      # no jax installed, or no backend
        print(f"GPU: not available ({err})")
        return
    print(f"GPU: {devices}")
    try:
        rows = bench_gpu(a.body, a.terrain, a.seconds, sizes)
    except Exception as err:      # e.g. a geom pair MJX does not support
        print(f"GPU run failed: {type(err).__name__}: {err}")
        return
    print(f"{'batch':>6} {'compile s':>10} {'episodes/s':>11} {'vs CPU':>7}")
    for n, compile_s, rate in rows:
        print(f"{n:6d} {compile_s:10.1f} {rate:11.1f} {rate / cpu:6.1f}x")


if __name__ == "__main__":
    main()

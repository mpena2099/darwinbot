"""
experiments/benchmark.py
------------------------
Find the fastest training device on this machine.

  python experiments/benchmark.py                          # measure what is here
  python experiments/benchmark.py --seconds 3 --workers 4
  python experiments/benchmark.py --cache /tmp/device.json

The unit of work is one CMA-ES generation of the default run (darwinbot/inner.py):
a 2-joint body, task steer (three commands), two randomised variants per
terrain and the four terrains of the terrain preset, so one terrain sees
popsize x variants x commands worlds and a generation costs
4 x that many episodes.

The CPU is timed with a multiprocessing Pool of workers, MuJoCo Warp on every
CUDA device (its warm-up run compiles, the next one is timed). Warp is only
recommended when it is at least 1.5x the CPU; its CPU device is never an
option, it is the CPU with more overhead. The result is cached so
`train.py --device auto` can pick it up without measuring again.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np

from darwinbot import morphology as mo
from darwinbot.controller import active_genes, random_controller
from darwinbot.evolve import BODY_MODEL
from darwinbot.fitness import commands
from darwinbot.sim import Episode
from darwinbot.terrain import PRESETS, flat

TASK = "steer"
VARIANTS = 2
TERRAIN_PRESET = "terrain"
N_TERRAINS = len(PRESETS[TERRAIN_PRESET])
WARP_SPEEDUP = 1.5              # warp must beat the CPU by this much
CPU_WALL = 2.0                 # s of simulation per CPU timing
CACHE_ENV = "DARWINBOT_CACHE"


def popsize(morph) -> int:
    """CMA-ES's default population for the body's active controller genes."""
    return 4 + int(3 * math.log(len(active_genes(morph))))


def worlds_per_terrain(morph=None) -> int:
    """Episodes on one terrain in one generation: candidates x variants x commands."""
    morph = morph or mo.template(2)
    return popsize(morph) * VARIANTS * len(commands(TASK))


def _cpu_chunk(job):
    seconds, n, seed = job
    ep = Episode(mo.template(2), flat(), seconds)
    ctrl = random_controller(np.random.default_rng(seed))
    for _ in range(n):
        ep.run(ctrl)
    return n


def cpu_episodes_per_second(seconds: float, workers: int, budget_s: float) -> float:
    """Episodes per second on a Pool of workers: chunks start at two episodes
    per worker and grow once towards CPU_WALL seconds of simulation."""
    limit = max(min(CPU_WALL, budget_s), 0.05)
    per, eps = 2, 0.0
    for _ in range(3):
        t = time.perf_counter()
        with Pool(workers) as pool:
            done = sum(pool.map(_cpu_chunk,
                                [(seconds, per, i) for i in range(workers)]))
        dt = time.perf_counter() - t
        eps = done / dt
        if dt >= limit:
            break
        per = min(64, max(per + 1, int(per * limit / dt)))
    return eps


def warp_devices():
    """The CUDA devices warp may train on, or None when warp is not installed."""
    try:
        import mujoco_warp  # noqa: F401
        import warp
    except ImportError:
        return None
    return [f"cuda:{i}" for i in range(warp.get_cuda_device_count())]


def warp_episodes_per_second(device: str, worlds: int, seconds: float) -> float:
    """Episodes per second on one CUDA device: one batch of worlds as warm-up
    (it compiles) and a second, timed one. Nominal physics, command 0."""
    from darwinbot.sim_warp import WarpBatchEpisode

    batch = WarpBatchEpisode(mo.template(2), flat(), seconds, device=device)
    ctrls = np.repeat(random_controller(np.random.default_rng(0))[None], worlds, axis=0)
    batch.run_batch(ctrls)                     # warm-up: captures the graph
    t = time.perf_counter()
    batch.run_batch(ctrls)
    return worlds / (time.perf_counter() - t)


def measure(seconds: float = 5.0, budget_s: float = 60.0, workers=None,
            verbose: bool = True) -> dict:
    """Time one generation on every training device here and recommend one.

    seconds is the simulated length of an episode (training default 5 s) and
    budget_s caps the CPU timing. The dict has device (cpu or warp), workers,
    cpu_eps, warp_eps (None without a CUDA GPU), gpu (None likewise), date
    (ISO) and body_model.
    """
    morph = mo.template(2)
    worlds = worlds_per_terrain(morph)
    workers = workers or os.cpu_count() or 1
    if verbose:
        print(f"[benchmark] one generation: {worlds} worlds per terrain, "
              f"{N_TERRAINS} terrains, {seconds:g} s episodes")

    cpu = cpu_episodes_per_second(seconds, workers, budget_s)
    rows = [(f"cpu ({workers} workers)", cpu)]
    rates, notes = {}, []
    devices = warp_devices()
    if devices is not None and not devices:
        notes.append("warp is installed but no CUDA GPU was found, so only the "
                     "CPU is an option (the warp CPU device is not used)")
    for dev in devices or ():
        try:
            rate = warp_episodes_per_second(dev, worlds, seconds)
        except Exception as err:          # unsupported model, out of memory...
            notes.append(f"warp on {dev} failed: {type(err).__name__}: {err}"[:200])
            continue
        rates[dev] = rate
        rows.append((f"warp {dev}", rate))

    gpu = max(rates, key=rates.get) if rates else None
    warp = rates[gpu] if gpu else None
    if gpu and warp < WARP_SPEEDUP * cpu:
        notes.append(f"{gpu} is only {warp / cpu:.2f}x the CPU, below the "
                     f"{WARP_SPEEDUP:g}x warp needs to be worth it")
    device = "warp" if gpu and warp >= WARP_SPEEDUP * cpu else "cpu"
    result = dict(device=device, workers=workers, cpu_eps=cpu, warp_eps=warp,
                  gpu=gpu, date=datetime.now().isoformat(timespec="seconds"),
                  body_model=BODY_MODEL)
    if verbose:
        print_report(result, rows, worlds, notes)
    return result


def print_report(result: dict, rows, worlds: int, notes=()) -> None:
    """The table, the recommendation and the command that uses it."""
    s_gen = N_TERRAINS * worlds
    print(f"\n{'option':<20}{'episodes/s':>12}{'s/generation':>14}{'vs CPU':>9}")
    for name, eps in rows:
        print(f"{name:<20}{eps:>12.1f}{s_gen / eps:>14.2f}{eps / rows[0][1]:>8.2f}x")
    if result["device"] == "warp":
        print(f"\nfastest: warp, {result['warp_eps'] / result['cpu_eps']:.2f}x the CPU")
        print("  python experiments/train.py --device warp")
    else:
        print(f"\nfastest: the CPU with {result['workers']} workers")
        print(f"  python experiments/train.py --device cpu --workers {result['workers']}")
    print("  (or let train.py reuse this choice: python experiments/train.py --device auto)")
    for note in notes:
        print(note)


def cache_path(path=None) -> str:
    """$DARWINBOT_CACHE (a directory) or ~/.cache/darwinbot, plus device.json."""
    if path:
        return os.fspath(path)
    root = os.environ.get(CACHE_ENV) or os.path.expanduser("~/.cache/darwinbot")
    return os.path.join(root, "device.json")


def save(result: dict, path=None) -> str:
    """Write the measurement as JSON; returns the file written."""
    path = cache_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return path


def load(path=None):
    """The cached measurement, or None when there is no readable one."""
    try:
        with open(cache_path(path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description="darwinbot: find the fastest training device")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="simulated seconds per episode")
    p.add_argument("--workers", type=int, default=None,
                   help="CPU processes (default: every core)")
    p.add_argument("--cache", default=None,
                   help=f"file to save the choice in (default: {cache_path()})")
    a = p.parse_args()
    path = save(measure(a.seconds, workers=a.workers), a.cache)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()

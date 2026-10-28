"""
tools/kaggle/bench_warp.py
--------------------------
Raw physics throughput of CPU MuJoCo (every core) against MuJoCo Warp on a GPU,
for DarwinBot bodies and growing numbers of parallel worlds. Answers whether
the inner CMA-ES loop, which evaluates many controllers on one body, would run
faster on a GPU.

  python tools/kaggle/bench_warp.py                        # every body, flat and rubble
  python tools/kaggle/bench_warp.py --worlds 64,256,1024 --seconds 2

Both sides step the same model with the same open-loop sine targets, updated
at 50 Hz as the real controller does; the CPG itself is left out on both sides,
so the numbers compare physics only. One episode of training is
EPISODE_SECONDS / timestep steps.

On Kaggle it runs as a GPU script kernel next to the code dataset (the
darwinbot.tar.gz made by `git archive`), and installs what it needs.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def kaggle_setup():
    """On Kaggle: install the dependencies and find the code dataset (Kaggle
    may have unpacked the tarball already)."""
    if not os.path.isdir("/kaggle/input"):
        return ROOT
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mujoco>=3.6.0",
                    "mujoco-warp", "warp-lang", "cma", "ribs", "numba"], check=True)
    tars = glob.glob("/kaggle/input/**/darwinbot.tar.gz", recursive=True)
    if tars:
        dest = "/tmp/darwinbot"
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["tar", "xzf", tars[0], "-C", dest], check=True)
        return dest
    return os.path.dirname(glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)[0])


sys.path.insert(0, kaggle_setup())

import mujoco   # noqa: E402
import numpy as np   # noqa: E402

from darwinbot import morphology as mo   # noqa: E402
from darwinbot.builder import build_mjcf   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402

BODIES = {"quadruped": lambda: mo.template(2),
          "hexapod": lambda: mo.template(3),
          "wheeled": lambda: mo.template(2, wheels=True)}
CONTROL_DT = 0.02      # s, 50 Hz as on the ESP32


def model_xml(body, terrain):
    t = get_terrains(terrain)[0]
    xml, _ = build_mjcf(BODIES[body](), t.assets, t.world, ground_z=t.spawn_z,
                        terrain_friction=t.friction)
    return xml


def targets(rng, nworld, nu):
    """Per world: amplitude, frequency and phase of a sine target per actuator."""
    return (rng.uniform(0.2, 0.8, (nworld, nu)), rng.uniform(0.5, 2.0, (nworld, 1)),
            rng.uniform(0, 2 * np.pi, (nworld, nu)))


def _cpu_chunk(job):
    xml, n, seconds, seed = job
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    amp, freq, ph = targets(np.random.default_rng(seed), n, m.nu)
    every = max(1, round(CONTROL_DT / m.opt.timestep))
    steps = 0
    for w in range(n):
        mujoco.mj_resetData(m, d)
        for k in range(int(seconds / m.opt.timestep)):
            if k % every == 0:
                d.ctrl[:] = amp[w] * np.sin(2 * np.pi * freq[w, 0] * d.time + ph[w])
            mujoco.mj_step(m, d)
            steps += 1
    return steps


def bench_cpu(xml, seconds, n, workers):
    jobs = [(xml, max(1, n // workers), seconds, i) for i in range(workers)]
    t = time.perf_counter()
    with Pool(workers) as pool:
        steps = sum(pool.map(_cpu_chunk, jobs))
    return steps / (time.perf_counter() - t)


def bench_warp(xml, seconds, nworld):
    import mujoco_warp as mjw
    import warp as wp
    mjm = mujoco.MjModel.from_xml_string(xml)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld, nconmax=64, njmax=512)
    amp, freq, ph = targets(np.random.default_rng(0), nworld, mjm.nu)
    every = max(1, round(CONTROL_DT / mjm.opt.timestep))
    with wp.ScopedCapture() as cap:          # one control period of physics
        for _ in range(every):
            mjw.step(m, d)
    periods = int(seconds / CONTROL_DT)

    def run():
        for k in range(periods):
            u = amp * np.sin(2 * np.pi * freq * (k * CONTROL_DT) + ph)
            wp.copy(d.ctrl, wp.array(u.astype(np.float32), dtype=float))
            wp.capture_launch(cap.graph)
        wp.synchronize()

    t = time.perf_counter()
    run()                                    # warm-up (first launches)
    first = time.perf_counter() - t
    t = time.perf_counter()
    run()
    dt = time.perf_counter() - t
    overflow = int(np.asarray(d.overflow.numpy()).any()) if hasattr(d, "overflow") else -1
    return nworld * periods * every / dt, first, overflow


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bodies", default=",".join(BODIES))
    p.add_argument("--terrains", default="flat,rubble")
    p.add_argument("--seconds", type=float, default=2.0, help="simulated seconds per world")
    p.add_argument("--worlds", default="64,256,1024,4096")
    p.add_argument("--cpu-worlds", type=int, default=16)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    try:
        import warp as wp
        wp.init()
        print(f"warp {wp.__version__}, devices {wp.get_cuda_devices()}")
    except Exception as err:
        print(f"warp not available: {err}")
        wp = None
    for body in a.bodies.split(","):
        for terrain in a.terrains.split(","):
            xml = model_xml(body, terrain)
            dt = mujoco.MjModel.from_xml_string(xml).opt.timestep
            cpu = bench_cpu(xml, a.seconds, a.cpu_worlds, a.workers)
            print(f"\n{body} on {terrain}: CPU {a.workers} cores {cpu:10.0f} steps/s "
                  f"= {cpu * dt / 5.0:6.1f} five-second episodes/s")
            if wp is None:
                continue
            for n in (int(s) for s in a.worlds.split(",")):
                try:
                    rate, first, overflow = bench_warp(xml, a.seconds, n)
                except Exception as err:      # unsupported feature, out of memory...
                    print(f"  warp {n:5d} worlds: failed: {type(err).__name__}: {err}"[:300])
                    break
                print(f"  warp {n:5d} worlds: {rate:10.0f} steps/s  {rate / cpu:6.2f}x CPU"
                      f"  (first run {first:5.1f} s, overflow {overflow})")


if __name__ == "__main__":
    main()

"""
experiments/validate.py
-----------------------
Check that an evolved gait survives a sim-to-real gap: run the elite on N
randomised models per terrain (friction, mass, servo strength, play, command
latency) and compare with the nominal model.

  python experiments/validate.py                           # best elite, 50 variants
  python experiments/validate.py --type 6legs-2wheels-14servos --n 20
  python experiments/validate.py --archive checkpoints/archive.pkl --workers 4

The archive's task and payload are used, so the robot is checked under the
conditions it was trained for (turn: |yaw| instead of dx).

Prints PASS (exit 0) when the mean distance loss across terrains is below 30%
else FAIL (exit 1).
"""

import argparse
import multiprocessing
import sys

import numpy as np

from darwinbot import morphology as morpho
from darwinbot import randomize
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from replay import load  # noqa: E402

MAX_LOSS = 0.30


def progress(m, task):
    """What the task is after: metres forward, or radians turned either way."""
    return abs(m.yaw) if task == "turn" else m.dx


def validate_terrain(task):
    """Pool worker: nominal and randomised runs of one terrain."""
    genes, ctrl, terrain, n, seconds, seed, payload, task = task
    ep = Episode(morpho.decode(genes), terrain, seconds, payload)
    nominal = ep.run(ctrl)
    rng = np.random.default_rng(seed)
    runs = [ep.run(ctrl, randomize.perturb(ep.model, rng)) for _ in range(n)]
    prog = np.array([progress(r, task) for r in runs])
    fell = np.array([r.fell or r.unstable for r in runs])
    base = progress(nominal, task)
    loss = 1.0 - prog.mean() / base if base > 1e-6 else float("nan")
    return terrain.name, base, float(prog.mean()), float(prog.min()), \
        float(fell.mean()), loss


def main():
    p = argparse.ArgumentParser(description="darwinbot: validate an elite on randomised models")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--type", dest="cell", default=None,
                   help="robot type, e.g. 6legs-2wheels-14servos (default: best overall)")
    p.add_argument("--n", type=int, default=50, help="variants per terrain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()

    archive, elite = load(a.archive, a.cell)
    print(morpho.summary(morpho.decode(elite.morph)))
    terrains = get_terrains(archive.terrains)
    tasks = [(elite.morph, elite.ctrl, t, a.n, archive.seconds, a.seed + i, archive.payload, archive.task)
             for i, t in enumerate(terrains)]
    if a.workers > 1:
        with multiprocessing.Pool(a.workers) as pool:
            rows = pool.map(validate_terrain, tasks)
    else:
        rows = [validate_terrain(t) for t in tasks]

    what, unit = ("|yaw|", "rad") if archive.task == "turn" else ("dx", "m")
    print(f"\n{a.n} variants per terrain, task {archive.task} ({what} in {unit})")
    print(f"terrain    nominal {what}  mean {what}  worst {what}  fall rate  loss")
    for name, nom, mean, worst, fall, loss in rows:
        print(f"{name:9s} {nom:10.2f} {mean:8.2f} {worst:9.2f} {fall:9.0%} {loss:6.0%}")
    losses = [r[5] for r in rows if not np.isnan(r[5])]
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    ok = bool(losses) and mean_loss < MAX_LOSS
    print(f"\nmean loss {mean_loss:.0%} (limit {MAX_LOSS:.0%}): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

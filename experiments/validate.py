"""
experiments/validate.py
-----------------------
Check that an evolved gait survives a sim-to-real gap: run the elite on N
randomised models per terrain (friction, mass, servo strength, play, command
latency) and compare with the nominal model.

  python experiments/validate.py                           # best elite, 50 variants
  python experiments/validate.py --type 6legs-2wheels-14servos --n 20
  python experiments/validate.py --archive checkpoints/archive.pkl --workers 4

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


def validate_terrain(task):
    """Pool worker: nominal and randomised runs of one terrain."""
    genes, ctrl, terrain, n, seconds, seed = task
    ep = Episode(morpho.decode(genes), terrain, seconds)
    nominal = ep.run(ctrl)
    rng = np.random.default_rng(seed)
    runs = [ep.run(ctrl, randomize.perturb(ep.model, rng)) for _ in range(n)]
    dx = np.array([r.dx for r in runs])
    fell = np.array([r.fell or r.unstable for r in runs])
    loss = 1.0 - dx.mean() / nominal.dx if nominal.dx > 1e-6 else float("nan")
    return terrain.name, nominal.dx, float(dx.mean()), float(dx.min()), \
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
    tasks = [(elite.morph, elite.ctrl, t, a.n, archive.seconds, a.seed + i)
             for i, t in enumerate(terrains)]
    if a.workers > 1:
        with multiprocessing.Pool(a.workers) as pool:
            rows = pool.map(validate_terrain, tasks)
    else:
        rows = [validate_terrain(t) for t in tasks]

    print(f"\n{a.n} variants per terrain")
    print("terrain    nominal dx  mean dx  worst dx  fall rate  loss")
    for name, nom, mean, worst, fall, loss in rows:
        print(f"{name:9s} {nom:10.2f} {mean:8.2f} {worst:9.2f} {fall:9.0%} {loss:6.0%}")
    losses = [r[5] for r in rows if not np.isnan(r[5])]
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    ok = bool(losses) and mean_loss < MAX_LOSS
    print(f"\nmean loss {mean_loss:.0%} (limit {MAX_LOSS:.0%}): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

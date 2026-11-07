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
conditions it was trained for (turn: |yaw| instead of dx). A steer robot is
checked under every steering command: straight (dx), left and right (radians
turned the way asked). A command the robot does not follow even on the nominal
model has no loss to measure and is left out of the mean.

Prints PASS (exit 0) when the mean loss across terrains and commands is below
30% else FAIL (exit 1).
"""

import argparse
import multiprocessing
import sys

import numpy as np

from darwinbot import morphology as morpho
from darwinbot import randomize
from darwinbot.fitness import commands
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from replay import load  # noqa: E402

MAX_LOSS = 0.30


COMMAND_NAMES = {0.0: "straight", -1.0: "left", 1.0: "right"}


def progress(m, task, command=0.0):
    """What the task is after: metres forward, radians turned either way
    (turn), or radians turned the way asked (a steering command)."""
    if task == "turn":
        return abs(m.yaw)
    return m.progress if command == 0.0 else -command * m.yaw


def validate_terrain(task):
    """Pool worker: nominal and randomised runs of one terrain and command.
    Every command of a terrain meets the same variants (same seed)."""
    genes, ctrl, terrain, n, seconds, seed, payload, task, command = task
    ep = Episode(morpho.decode(genes), terrain, seconds, payload)
    nominal = ep.run(ctrl, None, command)
    rng = np.random.default_rng(seed)
    runs = [ep.run(ctrl, randomize.perturb(ep.model, rng), command) for _ in range(n)]
    prog = np.array([progress(r, task, command) for r in runs])
    fell = np.array([r.fell or r.unstable for r in runs])
    base = progress(nominal, task, command)
    loss = 1.0 - prog.mean() / base if base > 1e-6 else float("nan")
    return terrain.name, COMMAND_NAMES.get(command, f"{command:+g}"), base, \
        float(prog.mean()), float(prog.min()), float(fell.mean()), loss


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
    tasks = [(elite.morph, elite.ctrl, t, a.n, archive.seconds, a.seed + i,
              archive.payload, archive.task, c)
             for i, t in enumerate(terrains) for c in commands(archive.task)]
    if a.workers > 1:
        with multiprocessing.Pool(a.workers) as pool:
            rows = pool.map(validate_terrain, tasks)
    else:
        rows = [validate_terrain(t) for t in tasks]

    units = {"turn": "|yaw| in rad", "steer": "straight: dx in m, left/right: "
             "yaw the way asked in rad"}.get(archive.task, "dx in m")
    print(f"\n{a.n} variants per terrain, task {archive.task} ({units})")
    print("terrain   command   nominal    mean   worst  fall rate  loss")
    for name, cmd, nom, mean, worst, fall, loss in rows:
        shown = "   -  (does not follow it)" if np.isnan(loss) else f"{loss:6.0%}"
        print(f"{name:9s} {cmd:8s} {nom:8.2f} {mean:7.2f} {worst:7.2f} {fall:9.0%} {shown}")
    losses = [r[6] for r in rows if not np.isnan(r[6])]
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    ok = bool(losses) and mean_loss < MAX_LOSS
    print(f"\nmean loss {mean_loss:.0%} (limit {MAX_LOSS:.0%}): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""
experiments/validate.py
-----------------------
Check that an evolved gait survives a sim-to-real gap: run the elite on N
randomised models per terrain (friction, mass, servo strength, play, command
latency) and compare with the nominal model.

  python experiments/validate.py                           # best elite, 50 variants
  python experiments/validate.py --type 6legs-2wheels-14servos --n 20
  python experiments/validate.py --archive checkpoints/archive.pkl --workers 4
  python experiments/validate.py --unseen                  # and the untrained grounds

The archive's task and payload are used, so the robot is checked under the
conditions it was trained for (turn: |yaw| instead of dx). A steer robot is
checked under every command: straight (dx), left and right (radians turned
the way asked) and half speed (dx).

A robot passes only when it is robust: mean loss below MAX_LOSS, every row's
5th-percentile loss below P5_LOSS and fall rate at most MAX_FALL, and every
terrain moving on the nominal model. A row with ~0 nominal progress has no
loss, but still counts for falls and its terrain still has to move: a ground
where the robot does not move at all is a failure.

--unseen also runs the terrains the archive was not trained on (random ones
with another seed), reported and judged separately; the last verdict line is
the trained-terrain one. Exit 0 only when every checked set passes.
"""

import argparse
import multiprocessing
import sys
from typing import List, NamedTuple, Tuple

import numpy as np

from darwinbot import morphology as morpho
from darwinbot import ablate, body_options, plugins, power, rules, randomize
from darwinbot.fitness import commands
from darwinbot.sim import Episode
from darwinbot.terrain import SEEDED, TERRAINS, get_terrains

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from replay import load  # noqa: E402

MAX_LOSS = 0.30     # mean loss over every row
P5_LOSS = 0.60      # loss at the 5th percentile of the variants, every row
MAX_FALL = 0.10     # share of the variants that fall, every row


COMMAND_NAMES = {(0.0, 1.0): "straight", (-1.0, 1.0): "left", (1.0, 1.0): "right",
                 (0.0, 0.5): "half speed"}


class Row(NamedTuple):
    """One (terrain, command) cell: nominal and randomised progress, in the
    task's units."""
    terrain: str
    command: str
    nominal: float
    mean: float
    p5: float
    worst: float
    fall: float

    @property
    def label(self) -> str:
        return f"{self.terrain}/{self.command}"

    def loss(self, value: float) -> float:
        return 1.0 - value / self.nominal if self.nominal > 1e-6 else float("nan")

    @property
    def mean_loss(self) -> float:
        return self.loss(self.mean)

    @property
    def p5_loss(self) -> float:
        return self.loss(self.p5)


def verdict(rows: List[Row]) -> Tuple[bool, List[str]]:
    """(ok, reasons) of a set of rows, with no I/O. Rows with ~0 nominal
    progress carry no loss; they are still judged on falls and on whether
    their terrain moves."""
    if not rows:
        return False, ["no rows"]
    reasons = []
    losses = [r.mean_loss for r in rows if not np.isnan(r.mean_loss)]
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    if not mean_loss < MAX_LOSS:
        reasons.append(f"mean loss {mean_loss:.0%} (limit {MAX_LOSS:.0%})")
    slow = [r.label for r in rows if not np.isnan(r.p5_loss) and not r.p5_loss < P5_LOSS]
    if slow:
        reasons.append(f"p5 loss >= {P5_LOSS:.0%}: " + ", ".join(slow))
    falling = [r.label for r in rows if r.fall > MAX_FALL]
    if falling:
        reasons.append(f"fall rate > {MAX_FALL:.0%}: " + ", ".join(falling))
    for terrain in dict.fromkeys(r.terrain for r in rows):
        if all(r.nominal <= 1e-6 for r in rows if r.terrain == terrain):
            reasons.append(f"no progress on {terrain}")
    return not reasons, reasons


def progress(m, task, command=(0.0, 1.0)):
    """What the task is after: metres forward, radians turned either way
    (turn), or radians turned the way asked (a steering command)."""
    if task == "turn":
        return abs(m.yaw)
    return m.progress if command[0] == 0.0 else -command[0] * m.yaw


def validate_terrain(task) -> Row:
    """Pool worker: nominal and randomised runs of one terrain and command.
    Every command of a terrain meets the same variants (same seed)."""
    genes, ctrl, terrain, n, seconds, seed, payload, task, command, settings = task
    ablate.set_active(settings[0])
    rules.set_active(settings[1])
    body_options.set_active(settings[2])
    ep = Episode(morpho.decode(genes), terrain, seconds, payload)
    nominal = ep.run(ctrl, None, *command)
    rng = np.random.default_rng(seed)
    runs = [ep.run(ctrl, randomize.perturb(ep.model, rng), *command) for _ in range(n)]
    prog = np.array([progress(r, task, command) for r in runs])
    fell = np.array([r.fell or r.unstable for r in runs])
    return Row(terrain.name, COMMAND_NAMES.get(tuple(command), str(command)),
               progress(nominal, task, command), float(prog.mean()),
               float(np.percentile(prog, 5)), float(prog.min()), float(fell.mean()))


def run_rows(elite, archive, terrains, n, seed, workers) -> List[Row]:
    tasks = [(elite.morph, elite.ctrl, t, n, archive.seconds, seed + i,
              archive.payload, archive.task, c,
              (archive.ablate, archive.rules, archive.body))
             for i, t in enumerate(terrains) for c in commands(archive.task)]
    if workers > 1:
        with multiprocessing.Pool(workers, plugins.load_all, (plugins.loaded(),)) as pool:
            return pool.map(validate_terrain, tasks)
    return [validate_terrain(t) for t in tasks]


def unseen_terrains(archive, seed):
    """Terrains the archive never trained on; the random ones get another seed
    so they are not the fixed default either."""
    seen = {t.name for t in get_terrains(archive.terrains)}
    return [TERRAINS[n](seed=(seed + 7919 * i) % 2**31) if n in SEEDED else TERRAINS[n]()
            for i, n in enumerate(n for n in TERRAINS if n not in seen)]


def shown(value: float) -> str:
    return "     -" if np.isnan(value) else f"{value:6.0%}"


def report(title: str, rows: List[Row]) -> bool:
    """Print a table and one line per failed criterion; return the verdict."""
    print(f"\n{title}")
    print(f"{'terrain':9s} {'command':10s} {'nominal':>7s} {'mean':>6s} {'p5':>6s} "
          f"{'min':>6s} {'fall':>5s} {'loss':>6s} {'p5 loss':>7s}")
    for r in rows:
        print(f"{r.terrain:9s} {r.command:10s} {r.nominal:7.2f} {r.mean:6.2f} "
              f"{r.p5:6.2f} {r.worst:6.2f} {r.fall:5.0%} {shown(r.mean_loss)} "
              f"{shown(r.p5_loss)}")
    ok, reasons = verdict(rows)
    for reason in reasons:
        print(f"  FAILED: {reason}")
    return ok


def main():
    p = argparse.ArgumentParser(description="darwinbot: validate an elite on randomised models")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains the archive was trained with")
    p.add_argument("--type", dest="cell", default=None,
                   help="robot type, e.g. 6legs-2wheels-14servos (default: best overall)")
    p.add_argument("--n", type=int, default=50, help="variants per terrain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--unseen", action="store_true",
                   help="also check the terrains the archive was not trained on")
    a = p.parse_args()

    archive, elite = load(a.archive, a.cell, a.plugin)
    morph = morpho.decode(elite.morph)
    print(morpho.summary(morph))
    units = {"turn": "|yaw| in rad", "steer": "straight: dx in m, left/right: "
             "yaw the way asked in rad"}.get(archive.task, "dx in m")

    rows = run_rows(elite, archive, get_terrains(archive.terrains), a.n, a.seed,
                    a.workers)
    ok = report(f"{a.n} variants per terrain, task {archive.task} ({units})", rows)
    unseen_ok = True

    if a.unseen:
        extra = unseen_terrains(archive, a.seed + 1000)
        if extra:
            unseen_ok = report("unseen terrains (not trained on): " +
                               ", ".join(t.name for t in extra),
                               run_rows(elite, archive, extra, a.n, a.seed + 1000,
                                        a.workers))
            print(f"unseen: {'PASS' if unseen_ok else 'FAIL'}")
        else:
            print("\nunseen: the archive trained on every terrain")

    flat = get_terrains("flat")[0]
    est = power.estimate(Episode(morph, flat, archive.seconds, archive.payload),
                         elite.ctrl, archive.seconds)
    print("\npower (nominal flat run, straight command):")
    for line in power.report_lines(est):
        print(f"  {line}")

    print(f"\ntrained terrains (mean loss < {MAX_LOSS:.0%}, p5 loss < {P5_LOSS:.0%}, "
          f"fall rate <= {MAX_FALL:.0%}): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok and unseen_ok else 1)


if __name__ == "__main__":
    main()

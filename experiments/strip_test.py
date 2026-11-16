"""
experiments/strip_test.py
-------------------------
A change of grip mid-stride: run a robot straight ahead across a short
low-friction strip (terrain `strip`) placed at several distances, on the
nominal model and on randomised ones, and report how often it stays upright
and how often it also walks one body length past the strip's far edge. For
evaluation only: it is not part of training or of the score.

  python experiments/strip_test.py --archive checkpoints/archive.pkl --type 8servos-mid-trot
  python experiments/strip_test.py --positions 0.3,0.5,0.8,1.2 --length 0.3 --mu 0.2 --n 20

The body length is twice the robot's reach (morphology.max_reach). A position
the robot does not reach on plain flat ground (its far edge plus a body length
is beyond the nominal flat progress) is "not reached", not a failure. PASS when
the pass share is at least --min-pass at every reached position; exit 0 on PASS,
1 on FAIL. Archive settings (task, payload, seconds, profile, rules) apply as in
validate.py.
"""

import argparse
import multiprocessing
import sys
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from darwinbot import morphology as morpho
from darwinbot import ablate, body_options, plugins, rules, randomize
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains, strip

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from replay import load  # noqa: E402

MIN_PASS = 0.8
FORWARD = (0.0, 1.0)    # command, speed


class Row(NamedTuple):
    """One strip position: `n` episodes (nominal and variants)."""
    at: float
    reached: bool
    n: int = 0
    upright: float = 0.0    # share that did not fall
    passed: float = 0.0     # share that did not fall and ended a body length past the strip


def passes(dx: float, fell: bool, far_edge: float, body: float) -> bool:
    return (not fell) and dx >= far_edge + body


def verdict(rows: List[Row], min_pass: float = MIN_PASS) -> Tuple[bool, List[float]]:
    """(ok, failed positions) with no I/O. Positions not reached are skipped;
    with none reached there is nothing to pass."""
    reached = [r for r in rows if r.reached]
    failed = [r.at for r in reached if r.passed < min_pass]
    return bool(reached) and not failed, failed


def _settings(settings):
    ablate.set_active(settings[0])
    rules.set_active(settings[1])
    body_options.set_active(settings[2])


def flat_progress(task) -> float:
    """Pool worker: nominal forward dx on plain flat ground."""
    genes, ctrl, seconds, payload, settings = task
    _settings(settings)
    ep = Episode(morpho.decode(genes), get_terrains("flat")[0], seconds, payload)
    return ep.run(ctrl, None, *FORWARD).dx


def run_position(task) -> Row:
    """Pool worker: nominal run and n variants with the strip at one place."""
    genes, ctrl, at, length, mu, far_edge, body, n, seconds, seed, payload, settings = task
    _settings(settings)
    ep = Episode(morpho.decode(genes), strip(at, length, mu), seconds, payload)
    rng = np.random.default_rng(seed)
    runs = [ep.run(ctrl, None, *FORWARD)]
    runs += [ep.run(ctrl, randomize.perturb(ep.model, rng), *FORWARD) for _ in range(n)]
    fell = [r.fell or r.unstable for r in runs]
    ok = [passes(r.dx, f, far_edge, body) for r, f in zip(runs, fell)]
    return Row(at, True, len(runs), 1.0 - float(np.mean(fell)), float(np.mean(ok)))


def report(rows: List[Row], min_pass: float = MIN_PASS) -> bool:
    print(f"{'at (m)':>7s} {'episodes':>8s} {'upright':>8s} {'pass':>6s}")
    for r in rows:
        if r.reached:
            print(f"{r.at:7.2f} {r.n:8d} {r.upright:8.0%} {r.passed:6.0%}")
        else:
            print(f"{r.at:7.2f} {'not reached':>24s}")
    ok, failed = verdict(rows, min_pass)
    if ok:
        print(f"PASS (pass share >= {min_pass:.0%} at every reached position)")
    elif not any(r.reached for r in rows):
        print("FAIL: no position reached on flat ground")
    else:
        print(f"FAIL at {', '.join(f'{x:g}' for x in failed)} m "
              f"(pass share < {min_pass:.0%})")
    return ok


def main():
    p = argparse.ArgumentParser(description="darwinbot: strip test, a change of grip mid-stride")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--plugin", default=None, metavar="FILE.py")
    p.add_argument("--type", dest="cell", default=None,
                   help="robot type (default: best overall)")
    p.add_argument("--positions", default="0.3,0.5,0.8,1.2",
                   help="metres from the spawn to the strip's near edge")
    p.add_argument("--length", type=float, default=0.3, help="strip length (m)")
    p.add_argument("--mu", type=float, default=0.2, help="friction of the strip")
    p.add_argument("--n", type=int, default=20, help="physics variants per position")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--min-pass", type=float, default=MIN_PASS)
    a = p.parse_args()

    positions = [float(x) for x in a.positions.split(",")]
    archive, elite = load(a.archive, a.cell, a.plugin)
    morph = morpho.decode(elite.morph)
    print(morpho.summary(morph))
    body = 2.0 * morph.max_reach
    settings = (archive.ablate, archive.rules, archive.body)
    common = (elite.morph, elite.ctrl)
    flat_task = (*common, archive.seconds, archive.payload, settings)
    tasks = [(*common, at, a.length, a.mu, at + a.length, body, a.n, archive.seconds,
              a.seed, archive.payload, settings) for at in positions]

    pool = (multiprocessing.Pool(a.workers, plugins.load_all, (plugins.loaded(),))
            if a.workers > 1 else None)
    try:
        flat_dx = pool.apply(flat_progress, (flat_task,)) if pool else flat_progress(flat_task)
        reach = [at + a.length + body <= flat_dx for at in positions]
        todo = [t for t, r in zip(tasks, reach) if r]
        done = iter(pool.map(run_position, todo) if pool else map(run_position, todo))
        rows = [next(done) if r else Row(at, False) for at, r in zip(positions, reach)]
    finally:
        if pool:
            pool.close()
    print(f"\nflat nominal dx {flat_dx:.2f} m, body length {body:.2f} m, strip "
          f"{a.length:g} m at mu {a.mu:g}, {a.n} variants per position")
    ok = report(rows, a.min_pass)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

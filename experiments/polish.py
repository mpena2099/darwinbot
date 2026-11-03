"""
Give chosen robots a longer controller search than evolution could afford.

    python experiments/polish.py --archive a.pkl --types t1,t2 --out polished.pkl
    python experiments/polish.py --archive a.pkl --per-gait roll,trot,walk --out polished.pkl

Training tunes a controller only long enough to judge a body, so an elite
usually enters the archive far below what its body allows. For each chosen
robot this runs a longer CMA-ES from the robot's own controller, on the same
terrains, physics variants, task and payload as its run, and writes a new
archive with only those robots, for replay, validate and export. --types picks
robots by name (as in replay.py --list); --per-gait picks the best elite of
each listed gait. A robot whose polished behaviour lands in a cell another
robot already took goes to polished-2.pkl (-3, ...), and a robot that did not
improve keeps its original controller.
"""

import argparse
import dataclasses
import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from darwinbot import morphology as morpho   # noqa: E402
from darwinbot.evolve import Archive, cell_of, scaled_budget, type_name   # noqa: E402
from darwinbot.inner import optimise_controller   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402
from replay import open_archive   # noqa: E402

GAITS = ("crawl", "walk", "trot", "belly", "roll")


def polish_elite(job):
    cell, elite, a, budget, sigma, seed = job
    m = morpho.decode(elite.morph)
    start = time.time()
    r = optimise_controller(m, get_terrains(a.terrains, seed=seed), x0=elite.ctrl,
                            budget=scaled_budget(budget, m), sigma0=sigma, seed=seed,
                            seconds=a.seconds, variants=a.variants, task=a.task,
                            payload=a.payload)
    minutes = (time.time() - start) / 60
    if r.fitness >= elite.fitness:
        e = dataclasses.replace(elite, ctrl=r.ctrl, fitness=r.fitness,
                                scores=list(r.scores), behaviour=r.behaviour)
        note = ""
    else:
        e, note = elite, ", kept the original controller"
    print(f"{type_name(elite.behaviour)} -> {type_name(e.behaviour)}  "
          f"{elite.fitness:.3f} -> {r.fitness:.3f}  {minutes:.1f} min{note}", flush=True)
    return cell, e


def picks(a, types, gaits):
    """Cells of the robots to polish: the named ones, or the best of each gait."""
    if types:
        return [a.find(t) for t in types]
    best = {}
    for c, e in a.cells.items():
        gait = type_name(e.behaviour).split("-")[2]
        if gait in gaits and (gait not in best or e.fitness > best[gait][1].fitness):
            best[gait] = (c, e)
    missing = [g for g in gaits if g not in best]
    if missing:
        sys.exit(f"no elite with gait {', '.join(missing)} in the archive")
    return [best[g][0] for g in gaits]


def _new_archive(a) -> Archive:
    return Archive(terrains=a.terrains, batch=a.batch, seconds=a.seconds,
                   variants=a.variants, task=a.task, payload=a.payload)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True)
    choose = p.add_mutually_exclusive_group(required=True)
    choose.add_argument("--types", help="comma-separated type names, as in replay.py --list")
    choose.add_argument("--per-gait", help="comma-separated gaits: " + ", ".join(GAITS))
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=1200,
                   help="CMA-ES evaluations per robot, before scaling")
    p.add_argument("--sigma", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; pick a new name")
    gaits = None
    if args.per_gait:
        gaits = args.per_gait.split(",")
        bad = [g for g in gaits if g not in GAITS]
        if bad:
            sys.exit(f"unknown gait(s) {', '.join(bad)}; pick from {', '.join(GAITS)}")
    a: Archive = open_archive(args.archive)
    cells = picks(a, args.types.split(",") if args.types else None, gaits)
    jobs = [(c, a.cells[c], a, args.budget, args.sigma, args.seed) for c in cells]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(polish_elite, jobs)
    else:
        results = [polish_elite(job) for job in jobs]

    # One archive per collision group: every polished robot is kept, and an
    # archive still holds at most one elite per cell.
    base, ext = os.path.splitext(args.out)
    groups = [[_new_archive(a), set(), args.out]]
    for cell, e in results:
        c = cell_of(e.behaviour)
        group = next((g for g in groups if c not in g[1]), None)
        if group is None:
            n = len(groups) + 1
            while os.path.exists(f"{base}-{n}{ext}"):
                n += 1
            group = [_new_archive(a), set(), f"{base}-{n}{ext}"]
            groups.append(group)
        group[1].add(c)
        group[0].insert(e)
        if group[2] != args.out:
            print(f"{type_name(e.behaviour)} shares a cell with another polished robot; "
                  f"saved to {group[2]}", flush=True)
    for arc, _, path in groups:
        with open(path, "wb") as f:
            pickle.dump(arc, f)
    print(f"saved {sum(len(arc.cells) for arc, _, _ in groups)} robots to "
          f"{', '.join(path for _, _, path in groups)}", flush=True)


if __name__ == "__main__":
    main()

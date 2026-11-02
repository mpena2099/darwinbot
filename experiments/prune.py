"""
Remove the parts a robot does not need, after training.

    python experiments/prune.py --archive checkpoints/final/archive.pkl --types a,b --out pruned.pkl

Evolution never pays for an extra part, so limbs that do nothing drift in. For
each chosen elite this tries to cut every leaf module in turn, gives the
smaller body a short controller search from its current controller, and keeps
the cut when the score stays within --tol of the body's score. It repeats
until no leaf can go. The result is a new archive with only the pruned robots,
for replay, validate and export.
"""

import argparse
import copy
import os
import pickle
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from darwinbot import morphology as morpho   # noqa: E402
from darwinbot.evolve import Archive, Elite, scaled_budget   # noqa: E402
from darwinbot.inner import optimise_controller   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402


def _leaves(m):
    parents = {n.parent for n in m.nodes.values()}
    return [i for i in m.nodes if i != 0 and i not in parents]


def prune_elite(job):
    name, elite, a, budget, tol, seed = job
    m = morpho.decode(elite.morph)
    ctrl, fit, scores, beh = elite.ctrl, elite.fitness, elite.scores, elite.behaviour
    floor = fit - tol * abs(fit)
    cut = []
    while True:
        best = None
        for leaf in _leaves(m):
            trial = copy.deepcopy(m)
            del trial.nodes[leaf]
            if trial.n_servos == 0:
                continue
            r = optimise_controller(trial, get_terrains(a.terrains, seed=seed), x0=ctrl,
                                    budget=scaled_budget(budget, trial), sigma0=0.12,
                                    seed=seed, seconds=a.seconds, variants=a.variants,
                                    task=a.task, payload=a.payload)
            if r.evals and r.fitness >= floor and (best is None or r.fitness > best[1].fitness):
                best = (leaf, r, trial)
        if best is None:
            break
        leaf, r, m = best
        cut.append(f"#{leaf}")
        ctrl, scores, beh = r.ctrl, r.scores, r.behaviour
        fit = r.fitness
        floor = max(floor, fit - tol * abs(fit))
    print(f"{name}: cut {len(cut)} parts {cut}, score {elite.fitness:.3f} -> {fit:.3f}",
          flush=True)
    return Elite(morpho.encode(m), ctrl, fit, scores, elite.born, beh)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True)
    p.add_argument("--types", required=True, help="comma-separated type names")
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=120, help="CMA-ES evaluations per cut, before scaling")
    p.add_argument("--tol", type=float, default=0.02, help="score share a cut may lose")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; pick a new name")
    with open(args.archive, "rb") as f:
        a: Archive = pickle.load(f)
    jobs = [(t, a.cells[a.find(t)], a, args.budget, args.tol, args.seed)
            for t in args.types.split(",")]
    with Pool(args.workers) as pool:
        elites = pool.map(prune_elite, jobs)
    out = Archive(terrains=a.terrains, seconds=a.seconds, variants=a.variants,
                  task=a.task, payload=a.payload)
    for e in elites:
        out.insert(e)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"saved {len(out.cells)} robots to {args.out}; names: {sorted(out.names().values())}")


if __name__ == "__main__":
    main()

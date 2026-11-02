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

import numpy as np   # noqa: E402
from darwinbot import morphology as morpho, randomize   # noqa: E402
from darwinbot.evolve import Archive, Elite, scaled_budget   # noqa: E402
from darwinbot.inner import evaluate, optimise_controller   # noqa: E402
from darwinbot.sim import Episode   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402


def _leaves(m):
    parents = {n.parent for n in m.nodes.values()}
    return [i for i in m.nodes if i != 0 and i not in parents]


def _score(m, ctrl, a, seed):
    """Score on the same terrains and physics variants every trial uses."""
    episodes = [Episode(m, t, a.seconds, a.payload) for t in get_terrains(a.terrains, seed=seed)]
    if not episodes[0].buildable:
        return None
    rng = np.random.default_rng(seed)
    var_sets = ([[randomize.perturb(ep.model, rng) for _ in range(a.variants)]
                 for ep in episodes] if a.variants > 0 else None)
    return evaluate(episodes, ctrl, var_sets, a.task)[0]


def prune_elite(job):
    name, elite, a, budget, tol, seed = job
    m = morpho.decode(elite.morph)
    ctrl, scores, beh = elite.ctrl, elite.scores, elite.behaviour
    fit = _score(m, ctrl, a, seed)       # the uncut body, judged like the cuts
    start = fit
    floor = fit - tol * abs(fit)
    cut = []
    while True:
        best = None
        for leaf in _leaves(m):
            trial = copy.deepcopy(m)
            del trial.nodes[leaf]
            if trial.n_servos == 0:
                continue
            quick = _score(trial, ctrl, a, seed)    # inherited controller first
            if quick is None or quick < floor - 0.1 * abs(floor):
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
    print(f"{name}: cut {len(cut)} parts {cut}, score {start:.3f} -> {fit:.3f}",
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
    for t, e in zip(args.types.split(","), elites):
        if not out.insert(e):
            print(f"warning: {t} landed in a cell already taken by a better pruned robot")
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"saved {len(out.cells)} robots to {args.out}; names: {sorted(out.names().values())}")


if __name__ == "__main__":
    main()

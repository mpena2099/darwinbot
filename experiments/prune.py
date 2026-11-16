"""
Remove the parts a robot does not need, after training.

    python experiments/prune.py --archive checkpoints/final/archive.pkl --types a,b --out pruned.pkl

Evolution never pays for an extra part, so limbs that do nothing drift in. For
each chosen elite this tries to cut every leaf module in turn, gives the
smaller body a short controller search from its current controller, and keeps
the cut when the score stays within --tol of the body's score. It repeats
until no leaf can go. The result is a new archive with only the pruned robots,
for replay, validate and export.

The leaf trials of one round are independent, so they run on a worker pool
(one task per leaf, every robot's leaves in the same pool); the accepted cut
is still the best one of the round, in leaf order, so the result does not
depend on --workers.
"""

import argparse
import copy
import os
import pickle
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np   # noqa: E402
from darwinbot import hardware, morphology as morpho, plugins, randomize   # noqa: E402
from darwinbot.evolve import Archive, Elite, load_archive, scaled_budget   # noqa: E402
from darwinbot.inner import evaluate, optimise_controller   # noqa: E402
from darwinbot.sim import Episode   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402

_ARCHIVE = None     # the archive of this process (set by _set_archive in each worker)


def _set_archive(archive: Archive) -> None:
    global _ARCHIVE
    _ARCHIVE = archive
    archive.activate()


def _init(loaded, archive: Archive) -> None:
    """Pool initializer: the plugins the parent loaded, then the archive's settings."""
    plugins.load_all(loaded)
    _set_archive(archive)


def _leaves(m):
    parents = {n.parent for n in m.nodes.values()}
    return [i for i in m.nodes if i != 0 and i not in parents]


def _episodes(m, a, seed):
    return [Episode(m, t, a.seconds, a.payload) for t in get_terrains(a.terrains, seed=seed)]


def _score(episodes, ctrl, a, seed):
    """(score, per-terrain scores) on the same terrains and physics variants every trial uses."""
    rng = np.random.default_rng(seed)
    var_sets = ([[randomize.perturb(ep.model, rng) for _ in range(a.variants)]
                 for ep in episodes] if a.variants > 0 else None)
    fit, scores, _ = evaluate(episodes, ctrl, var_sets, a.task)
    return fit, scores


def _trial(job):
    """Pool worker: judge one cut and, when it is close enough, search its controller.
    None when the body does not build or the inherited controller already falls short."""
    leaf, trial, ctrl, floor, budget, seed = job
    a = _ARCHIVE
    episodes = _episodes(trial, a, seed)     # built once: the quick trial and the search share them
    if not episodes[0].buildable:
        return None
    quick, _ = _score(episodes, ctrl, a, seed)    # inherited controller first
    if quick < floor - 0.1 * abs(floor):
        return None
    return optimise_controller(trial, get_terrains(a.terrains, seed=seed), x0=ctrl,
                               budget=scaled_budget(budget, trial), sigma0=0.12,
                               seed=seed, seconds=a.seconds, variants=a.variants,
                               task=a.task, payload=a.payload, episodes=episodes)


def _prune_steps(name, elite, a, budget, tol, seed):
    """One robot's search as a generator: it yields the leaf trials of a round, takes
    back their results, and returns the robot's Elite. Fitness, scores and behaviour
    of the result always come from one evaluation."""
    m = morpho.decode(elite.morph)
    ctrl, beh = elite.ctrl, elite.behaviour
    episodes = _episodes(m, a, seed)
    if not episodes[0].buildable:
        print(f"{name}: does not build, kept as it is", flush=True)
        return elite
    fit, scores = _score(episodes, ctrl, a, seed)    # the uncut body, judged like the cuts
    start = fit
    floor = fit - tol * abs(fit)
    cut = []
    while True:
        leaves = []
        for leaf in _leaves(m):
            trial = copy.deepcopy(m)
            del trial.nodes[leaf]
            if trial.n_servos:
                leaves.append((leaf, trial))
        if not leaves:
            break
        results = yield [(leaf, trial, ctrl, floor, budget, seed) for leaf, trial in leaves]
        best = None
        for (leaf, trial), r in zip(leaves, results):
            if r is not None and r.evals and r.fitness >= floor and \
                    (best is None or r.fitness > best[1].fitness):
                best = (leaf, r, trial)
        if best is None:
            break
        leaf, r, m = best
        cut.append(f"#{leaf}")
        ctrl, scores, beh, fit = r.ctrl, r.scores, r.behaviour, r.fitness
        floor = max(floor, fit - tol * abs(fit))
    morph = morpho.encode(m) if cut else elite.morph
    print(f"{name}: cut {len(cut)} parts {cut}, score {start:.3f} -> {fit:.3f}", flush=True)
    return Elite(morph, ctrl, fit, scores, elite.born, beh)


def _step(name, gen, send, active, done):
    try:
        active[name] = (gen, gen.send(send))
    except StopIteration as stop:
        done[name] = stop.value


def _rounds(a, names, budget, tol, seed, map_):
    """Run every robot's generator in lock-step: each round, the leaf trials of all
    robots that are still searching go to map_ together."""
    active, done = {}, {}
    for name in names:
        _step(name, _prune_steps(name, a.cells[a.find(name)], a, budget, tol, seed),
              None, active, done)
    while active:
        current = list(active.items())
        active.clear()
        jobs = [job for _, (_, tasks) in current for job in tasks]
        out = map_(jobs)
        pos = 0
        for name, (gen, tasks) in current:
            _step(name, gen, out[pos:pos + len(tasks)], active, done)
            pos += len(tasks)
    return [done[n] for n in names]


def prune_all(a: Archive, names, budget, tol, seed, workers):
    """The pruned Elite of each named robot, in the order given."""
    if workers > 1:
        with Pool(workers, _init, (plugins.loaded(), a)) as pool:
            return _rounds(a, names, budget, tol, seed,
                           lambda jobs: pool.map(_trial, jobs, chunksize=1))
    _set_archive(a)
    return _rounds(a, names, budget, tol, seed, lambda jobs: [_trial(j) for j in jobs])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True)
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains the archive was trained with")
    p.add_argument("--types", required=True, help="comma-separated type names")
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=120, help="CMA-ES evaluations per cut, before scaling")
    p.add_argument("--tol", type=float, default=0.02, help="score share a cut may lose")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                   help="worker processes for the leaf trials (default: all CPUs)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; pick a new name")
    a: Archive = load_archive(args.archive)
    plugins.require(a, args.plugin)
    hardware.require(a.profile)
    a.activate()
    names = args.types.split(",")
    elites = prune_all(a, names, args.budget, args.tol, args.seed, args.workers)
    out = a.like()
    for t, e in zip(names, elites):
        if not out.insert(e):
            print(f"warning: {t} landed in a cell already taken by a better pruned robot")
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"saved {len(out.cells)} robots to {args.out}; names: {sorted(out.names().values())}")


if __name__ == "__main__":
    main()

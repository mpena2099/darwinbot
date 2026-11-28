"""
Benchmark inner-loop CMA-ES strategies on the robots of an archive (numpy only).

    python tools/speed/bench_inner.py --archive checkpoints/steer/archive.pkl \
        --pairs 8 --seeds 2 --budget 120 --out bench/

A pair is an elite and a mutated child of it: the child's controller is
optimised from the parent's controller under every strategy and seed, and the
best fitness so far is recorded after each evaluation. Strategies are the
`strategy` argument of inner.optimise_controller: baseline (None), diagonal,
focused and diagonal+focused. `--variant-mode` is inner.optimise_controller's
`variant_mode`, and every run is named `strategy/mode`. Output: DIR/runs.npz
(curves, padded with NaN, plus pair/strategy/seed indices) and DIR/summary.md.
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from darwinbot import hardware, morphology as morpho, plugins   # noqa: E402
from darwinbot.evolve import load_archive   # noqa: E402
from darwinbot.inner import optimise_controller   # noqa: E402
from darwinbot.sim import Episode   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402

SIGMA0 = 0.12
STRATEGIES = {"baseline": None, "diagonal": "diagonal", "focused": "focused",
              "diagonal+focused": "diagonal+focused"}


def make_pairs(archive, n: int, rng: np.random.Generator, ground):
    """n (child genes, parent genes, parent ctrl) triples; unbuildable children
    are redrawn, and the loop stops early rather than spin."""
    cells = sorted(archive.cells)
    if not cells:
        raise SystemExit("the archive has no elites to mutate")
    elites = [archive.cells[c] for c in cells]
    pairs = []
    for _ in range(20 * n):
        if len(pairs) == n:
            break
        e = elites[int(rng.integers(len(elites)))]
        child = morpho.mutate(e.morph, rng)
        ep = Episode(morpho.decode(child), ground, archive.seconds, archive.payload)
        if ep.buildable:
            pairs.append((child, e.morph, e.ctrl))
    if not pairs:
        raise SystemExit("no buildable child from the archive's elites")
    return pairs


def _run(job):
    (pair, strategy, seed, child, parent, x0, terrains, seconds, variants,
     task, payload, budget, variant_mode) = job
    curve = []
    res = optimise_controller(
        morpho.decode(child), get_terrains(terrains), x0=x0, budget=budget,
        sigma0=SIGMA0, seed=seed, seconds=seconds, variants=variants, task=task,
        payload=payload, strategy=STRATEGIES[strategy], parent_genes=parent,
        on_eval=curve.append, variant_mode=variant_mode)
    return pair, f"{strategy}/{variant_mode}", seed, curve, float(res.fitness)


def _best_at(curve: np.ndarray, frac: float, budget: int) -> float:
    """Best so far at a fraction of the budget; a run that stopped early keeps
    its last value. 100% uses the last point, which for "first" is the full
    re-score that follows the cheap loop."""
    n = int(np.isfinite(curve).sum())
    k = min(max(int(np.ceil(frac * budget)) - 1, 0), n - 1) if frac < 1.0 else n - 1
    return float(np.fmax.accumulate(curve)[k])


def _bootstrap(d: np.ndarray, rng: np.random.Generator, n: int = 2000):
    """90% interval of the mean of d, resampling the paired differences."""
    if len(d) < 2:
        return float("nan"), float("nan")
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [5, 95])
    return float(lo), float(hi)


def summary(results, names, budget: int, rng_seed: int) -> str:
    best = {}
    for pair, strategy, seed, curve, _ in results:
        row = np.asarray(curve, float)
        best[(strategy, pair, seed)] = [_best_at(row, f, budget)
                                        for f in (0.25, 0.5, 1.0)]
    rng = np.random.default_rng(rng_seed)
    base = next((n for n in names if n.split("/")[0] == "baseline"), None)
    rows = []
    for name in names:
        keys = sorted(k for k in best if k[0] == name)
        means = [float(np.mean([best[k][i] for k in keys])) for i in range(3)]
        diff, wtl = "-", "-"
        if name != base and base and any(k[0] == base for k in best):
            d = np.array([best[k][2] - best[(base, k[1], k[2])][2]
                          for k in keys])
            lo, hi = _bootstrap(d, rng)
            diff = f"{d.mean():+.4g} [{lo:+.4g}, {hi:+.4g}]"
            wins, ties = int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum())
            wtl = f"{wins}/{ties}/{len(d) - wins - ties}"
        rows.append([name, len(keys)] + [f"{m:.4f}" for m in means] + [diff, wtl])
    head = ["# Inner-loop strategy benchmark", "",
            f"{len(results)} runs, budget {budget} evaluations, sigma0 {SIGMA0}. "
            "Best-so-far fitness at 25/50/100% of the budget, mean over the runs of "
            "each strategy; the paired difference against baseline is at 100% with a "
            "bootstrap 90% interval over (pair, seed), and wins/ties/losses.", ""]
    table = ["| strategy | runs | 25% | 50% | 100% | vs baseline (90% CI) | W/T/L |",
             "|---|---|---|---|---|---|---|"]
    table += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(head + table) + "\n"


def save(out: str, results, names, budget: int, rng_seed: int) -> None:
    width = max(len(r[3]) for r in results)
    curves = np.full((len(results), width), np.nan)
    for row, r in enumerate(results):
        curves[row, :len(r[3])] = r[3]
    index = {s: i for i, s in enumerate(names)}
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(
        os.path.join(out, "runs.npz"), curves=curves,
        pair=np.array([r[0] for r in results], int),
        strategy=np.array([index[r[1]] for r in results], int),
        seed=np.array([r[2] for r in results], int),
        final=np.array([r[4] for r in results], float),
        strategy_names=np.array(names))
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(summary(results, names, budget, rng_seed))


def _parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--variant-mode", choices=("all", "first"), default="all",
                    help="first: one variant per terrain in the loop, one full re-score after")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--plugin", default=None, metavar="FILE.py",
                    help="external tasks and terrains the archive was trained with")
    return ap


def main():
    a = _parser().parse_args()
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategy {unknown[0]!r}; choose from {list(STRATEGIES)}")
    names = [f"{s}/{a.variant_mode}" for s in strategies]
    archive = load_archive(a.archive)
    plugins.require(archive, a.plugin)
    hardware.require(archive.profile)
    archive.activate()
    rng = np.random.default_rng(a.rng_seed)
    pairs = make_pairs(archive, a.pairs, rng, get_terrains(archive.terrains)[0])
    jobs = [(i, s, seed, child, parent, x0, archive.terrains, archive.seconds,
             archive.variants, archive.task, archive.payload, a.budget,
             a.variant_mode)
            for i, (child, parent, x0) in enumerate(pairs)
            for s in strategies for seed in range(a.seeds)]
    with Pool(a.workers, plugins.load_all, (plugins.loaded(),)) as pool:
        results = pool.map(_run, jobs)
    save(a.out, results, names, a.budget, a.rng_seed)
    print(f"{len(results)} runs over {len(pairs)} pairs: {a.out}/summary.md")


if __name__ == "__main__":
    main()

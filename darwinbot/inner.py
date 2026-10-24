"""
inner.py
--------
Inner loop: given a fixed body, find a good CPG controller with CMA-ES.

Separating the two levels fixes the core weakness of plain co-evolution: a
promising body used to die young because its random controller was bad.
Here every body is judged by (roughly) the best gait it can learn.

A body that is not buildable (legs overlap at spawn) gets UNSTABLE_SCORE
without any simulation or CMA-ES run.

Only the controller genes that affect this body are optimised; the rest are
carried over unchanged from x0 so they stay available to descendants that
grow the missing pair or wheel back.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cma
import numpy as np

from darwinbot import randomize

from darwinbot.controller import CTRL_LENGTH, active_genes, random_controller
from darwinbot.fitness import UNSTABLE_SCORE, aggregate, score
from darwinbot.morphology import Morphology
from darwinbot.sim import EPISODE_SECONDS, Episode, Metrics
from darwinbot.terrain import Terrain


@dataclass
class ControllerResult:
    ctrl:    np.ndarray
    fitness: float
    scores:  List[float]     # per terrain, same order as the terrain list
    metrics: List[Metrics]
    evals:   int


def evaluate(episodes: Sequence[Episode], ctrl: np.ndarray, variants=None):
    """
    variants: None, or one list of randomize.Variant per episode. The terrain
    score is then the mean over its variants (metrics keep the first one).
    """
    if not variants:
        metrics = [ep.run(ctrl) for ep in episodes]
        scores  = [score(m) for m in metrics]
        return aggregate(scores), scores, metrics
    metrics, scores = [], []
    for ep, vs in zip(episodes, variants):
        ms = [ep.run(ctrl, v) for v in vs]
        metrics.append(ms[0])
        scores.append(float(np.mean([score(m) for m in ms])))
    return aggregate(scores), scores, metrics


def _unbuildable(ctrl, terrains, mass) -> ControllerResult:
    m = Metrics(0.0, 0.0, 0.0, False, True, 0.0, mass)
    return ControllerResult(ctrl.copy(), UNSTABLE_SCORE,
                            [UNSTABLE_SCORE] * len(terrains),
                            [m] * len(terrains), evals=0)


def optimise_controller(
    morph: Morphology,
    terrains: Sequence[Terrain],
    x0: Optional[np.ndarray] = None,
    budget: int = 400,
    sigma0: float = 0.25,
    seed: int = 0,
    seconds: float = EPISODE_SECONDS,
    device: str = "cpu",
    popsize: Optional[int] = None,
    variants: int = 0,
) -> ControllerResult:
    """
    variants > 0: every candidate is scored on that many randomised models per
    terrain (domain randomisation, CPU only). They are drawn once per body from
    its seed, so all candidates face the same ones (common random numbers).
    """
    if variants > 0 and device == "gpu":
        raise ValueError("randomisation is CPU only; use --variants 0 with --device gpu")
    rng = np.random.default_rng(seed)
    base = np.array(x0, dtype=float) if x0 is not None else random_controller(rng)
    assert base.shape == (CTRL_LENGTH,)
    active = active_genes(morph)

    def full(z):
        c = base.copy()
        c[active] = z
        return c

    if device == "gpu":
        from darwinbot.sim_gpu import BatchEpisode
        batches = [BatchEpisode(morph, t, seconds) for t in terrains]
        if not batches[0].buildable:
            return _unbuildable(base, terrains, batches[0].mass)

        def evaluate_batch(ctrls):
            per_terrain = [b.run_batch(ctrls) for b in batches]
            out = []
            for i in range(len(ctrls)):
                metrics = [p[i] for p in per_terrain]
                scores = [score(m) for m in metrics]
                out.append((aggregate(scores), scores, metrics))
            return out

        def run_one(c, n: int = 1):
            # x0 goes through the population-sized batch so the first JIT
            # compile is reused by the whole loop.
            return evaluate_batch(np.repeat(c[None], n, axis=0))[0]

        # The starting point counts: an inherited controller that already works
        # must not be lost just because CMA-ES never resamples it exactly.
        fit, scores, metrics = run_one(base, popsize or 1)
    else:
        episodes = [Episode(morph, t, seconds) for t in terrains]
        if not episodes[0].buildable:
            return _unbuildable(base, terrains, episodes[0].mass)

        var_sets = None
        if variants > 0:
            vrng = np.random.default_rng([seed, 1])    # leaves rng untouched
            var_sets = [[randomize.perturb(ep.model, vrng) for _ in range(variants)]
                        for ep in episodes]

        def run_one(c, n: int = 1):
            return evaluate(episodes, c, var_sets)

        fit, scores, metrics = run_one(base)
    best = ControllerResult(base.copy(), fit, scores, metrics, evals=1)

    opts = {"bounds": [0.0, 1.0], "seed": int(rng.integers(1, 2**31 - 1)),
            "maxfevals": max(budget - 1, 1), "verbose": -9}
    if popsize is not None:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(base[active], sigma0, opts)

    evals = 1
    while not es.stop():
        candidates = es.ask()
        losses = []
        for z in candidates:
            c = full(np.asarray(z))
            fit, scores, metrics = run_one(c)
            evals += 1
            losses.append(-fit)
            if fit > best.fitness:
                best = ControllerResult(c, fit, scores, metrics, evals)
        es.tell(candidates, losses)

    best.evals = evals
    return best

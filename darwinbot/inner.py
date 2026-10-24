"""
inner.py
--------
Inner loop: given a fixed body, find a good CPG controller with CMA-ES.

Separating the two levels fixes the core weakness of plain co-evolution: a
promising body used to die young because its random controller was bad.
Here every body is judged by (roughly) the best gait it can learn.

Only the controller genes that affect this body are optimised; the rest are
carried over unchanged from x0 so they stay available to descendants that
grow the missing pair or wheel back.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cma
import numpy as np

from darwinbot.controller import CTRL_LENGTH, active_genes, random_controller
from darwinbot.fitness import aggregate, score
from darwinbot.morphology import Morphology
from darwinbot.sim import Episode, Metrics
from darwinbot.terrain import Terrain


@dataclass
class ControllerResult:
    ctrl:    np.ndarray
    fitness: float
    scores:  List[float]     # per terrain, same order as the terrain list
    metrics: List[Metrics]
    evals:   int


def evaluate(episodes: Sequence[Episode], ctrl: np.ndarray):
    metrics = [ep.run(ctrl) for ep in episodes]
    scores  = [score(m) for m in metrics]
    return aggregate(scores), scores, metrics


def optimise_controller(
    morph: Morphology,
    terrains: Sequence[Terrain],
    x0: Optional[np.ndarray] = None,
    budget: int = 400,
    sigma0: float = 0.25,
    seed: int = 0,
) -> ControllerResult:
    rng = np.random.default_rng(seed)
    episodes = [Episode(morph, t) for t in terrains]
    base = np.array(x0, dtype=float) if x0 is not None else random_controller(rng)
    assert base.shape == (CTRL_LENGTH,)
    active = active_genes(morph)

    def full(z):
        c = base.copy()
        c[active] = z
        return c

    # The starting point counts: an inherited controller that already works
    # must not be lost just because CMA-ES never resamples it exactly.
    fit, scores, metrics = evaluate(episodes, base)
    best = ControllerResult(base.copy(), fit, scores, metrics, evals=1)

    es = cma.CMAEvolutionStrategy(
        base[active], sigma0,
        {"bounds": [0.0, 1.0], "seed": int(rng.integers(1, 2**31 - 1)),
         "maxfevals": max(budget - 1, 1), "verbose": -9},
    )
    evals = 1
    while not es.stop():
        candidates = es.ask()
        losses = []
        for z in candidates:
            c = full(np.asarray(z))
            fit, scores, metrics = evaluate(episodes, c)
            evals += 1
            losses.append(-fit)
            if fit > best.fitness:
                best = ControllerResult(c, fit, scores, metrics, evals)
        es.tell(candidates, losses)

    best.evals = evals
    return best

"""
inner.py
--------
Inner loop: given a fixed body, find a good CPG controller with CMA-ES.

Separating the two levels fixes the core weakness of plain co-evolution: a
promising body used to die young because its random controller was bad.
Here every body is judged by (roughly) the best gait it can learn.

A body that is not buildable (legs overlap at spawn) gets UNSTABLE_SCORE
without any simulation or CMA-ES run.

The body's behaviour vector (duty, height, servo count, wheel share, cost of
transport) comes
from the nominal flat-terrain episode of its best controller, never from a
randomised variant.

Only the controller genes that affect this body are optimised; the rest are
carried over unchanged from x0 so they stay available to descendants that
grow the missing pair or wheel back.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import cma
import numpy as np

from darwinbot import randomize

from darwinbot.controller import CTRL_LENGTH, active_genes, random_controller
from darwinbot.fitness import UNSTABLE_SCORE, aggregate, commands, score, task_score
from darwinbot.morphology import Morphology
from darwinbot.sim import COT_MAX, EPISODE_SECONDS, Episode, Metrics, cost_of_transport
from darwinbot.terrain import Terrain, flat


@dataclass
class ControllerResult:
    ctrl:    np.ndarray
    fitness: float
    scores:  List[float]     # per terrain, same order as the terrain list
    metrics: List[Metrics]
    evals:   int
    behaviour: Optional[np.ndarray] = None   # (duty, height, servos, wheel share, cot)


def evaluate(episodes: Sequence[Episode], ctrl: np.ndarray, variants=None,
             task: str = "forward"):
    """
    variants: None, or one list of randomize.Variant per episode. The terrain
    score is then the mean over its variants (metrics keep the first one).
    """
    cmds = commands(task)
    if not variants:
        runs = [[ep.run(ctrl, None, c) for c in cmds] for ep in episodes]
        scores = [task_score(ms, task) for ms in runs]
        return aggregate(scores), scores, [ms[0] for ms in runs]
    metrics, scores = [], []
    for ep, vs in zip(episodes, variants):
        runs = [[ep.run(ctrl, v, c) for c in cmds] for v in vs]
        metrics.append(runs[0][0])
        scores.append(float(np.mean([task_score(ms, task) for ms in runs])))
    return aggregate(scores), scores, metrics


def behaviour_vector(morph: Morphology, m: Optional[Metrics],
                     task: str = "forward") -> np.ndarray:
    """(duty, height, servos, wheel share, cost of transport, legs) of a body
    from its nominal flat episode. A body never simulated, or whose episode blew up or
    gave non-finite numbers, gets the same fallback vector (never NaN)."""
    fallback = np.array([0.0, 0.0, float(morph.n_servos), 0.0, COT_MAX,
                         float(morph.n_legs)])
    if m is None or m.unstable:
        return fallback
    vec = np.array([m.duty, m.height, float(morph.n_servos), m.wheel_share,
                    cost_of_transport(m, task), float(morph.n_legs)])
    return vec if np.isfinite(vec).all() else fallback


def _unbuildable(morph, ctrl, terrains, mass) -> ControllerResult:
    m = Metrics(0.0, 0.0, 0.0, False, True, 0.0, mass)
    return ControllerResult(ctrl.copy(), UNSTABLE_SCORE,
                            [UNSTABLE_SCORE] * len(terrains),
                            [m] * len(terrains), evals=0,
                            behaviour=behaviour_vector(morph, None))


def cpu_evaluator(morph: Morphology, terrains: Sequence[Terrain], seconds: float,
                  payload: float, variants: int, seed: int, task: str):
    """Episodes of a body and the function that scores one full controller on
    them; the physics variants come from the seed alone, so every process that
    builds this with the same arguments scores a controller the same way."""
    episodes = [Episode(morph, t, seconds, payload) for t in terrains]
    var_sets = None
    if variants > 0 and episodes[0].buildable:
        vrng = np.random.default_rng([seed, 1])
        var_sets = [[randomize.perturb(ep.model, vrng) for _ in range(variants)]
                    for ep in episodes]
    return episodes, lambda c: evaluate(episodes, c, var_sets, task)


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
    task: str = "forward",
    payload: float = 0.0,
    map_many: Optional[Callable] = None,
) -> ControllerResult:
    """
    variants > 0: every candidate is scored on that many randomised models per
    terrain (domain randomisation; not on device "gpu"). They are drawn once
    per body from its seed, so all candidates face the same ones (common
    random numbers).
    task picks the score (fitness.TASKS); payload is extra torso mass in kg.
    The behaviour descriptor is the same for every task.

    device "warp" searches the controllers with MuJoCo Warp, one batch per
    terrain for the whole generation, and re-scores the winner on the CPU: the
    GPU only searches, what enters the archive is CPU-scored.

    map_many (CPU only): scores a list of full controllers, for example on a
    process pool whose workers built the same cpu_evaluator; the result does
    not change, only the wall time.
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
        batches = [BatchEpisode(morph, t, seconds, payload) for t in terrains]
        if not batches[0].buildable:
            return _unbuildable(morph, base, terrains, batches[0].mass)

        def evaluate_batch(ctrls):
            cmds = commands(task)
            per_terrain = [[b.run_batch(ctrls, c) for c in cmds] for b in batches]
            out = []
            for i in range(len(ctrls)):
                metrics = [p[0][i] for p in per_terrain]
                scores = [task_score([r[i] for r in p], task) for p in per_terrain]
                out.append((aggregate(scores), scores, metrics))
            return out

        def run_one(c, n: int = 1):
            # x0 goes through the population-sized batch so the first JIT
            # compile is reused by the whole loop.
            return evaluate_batch(np.repeat(c[None], n, axis=0))[0]

        def run_many(cs):
            # one generation = one batch on the device
            return evaluate_batch(np.stack(cs))

        # The starting point counts: an inherited controller that already works
        # must not be lost just because CMA-ES never resamples it exactly.
        fit, scores, metrics = run_one(base, popsize or 1)
    elif device == "warp":
        from darwinbot.sim_warp import WarpBatchEpisode
        batches = [WarpBatchEpisode(morph, t, seconds, payload) for t in terrains]
        if not batches[0].buildable:
            return _unbuildable(morph, base, terrains, batches[0].mass)

        # The same variants as the CPU branch, drawn from the same generator:
        # both devices then face the same randomised models.
        var_sets = None
        if variants > 0:
            vrng = np.random.default_rng([seed, 1])    # leaves rng untouched
            var_sets = [[randomize.perturb(b.episode.model, vrng)
                         for _ in range(variants)] for b in batches]

        def evaluate_batch(ctrls):
            """One world per (controller, variant, command) and one run_batch per
            terrain, so a whole generation is len(terrains) launches."""
            cmds = commands(task)
            nv, ncmd = max(variants, 1), len(cmds)
            per_terrain = []
            for i, b in enumerate(batches):
                vs = var_sets[i] if var_sets else [None]
                worlds = [(c, v, float(cmd)) for c in ctrls for v in vs
                          for cmd in cmds]
                per_terrain.append(b.start(      # the terrains run concurrently
                    np.stack([w[0] for w in worlds]),
                    np.array([w[2] for w in worlds]),
                    None if var_sets is None else [w[1] for w in worlds]))
            per_terrain = [b.finish(t) for b, t in zip(batches, per_terrain)]
            out = []
            for i in range(len(ctrls)):
                scores, metrics = [], []
                for p in per_terrain:          # rebuild what evaluate() returns
                    start = i * nv * ncmd
                    runs = [p[start + j * ncmd:start + (j + 1) * ncmd]
                            for j in range(nv)]
                    scores.append(float(np.mean([task_score(ms, task)
                                                 for ms in runs])))
                    metrics.append(runs[0][0])        # variant 0, first command
                out.append((aggregate(scores), scores, metrics))
            return out

        def run_one(c, n: int = 1):
            return evaluate_batch(np.repeat(c[None], n, axis=0))[0]

        def run_many(cs):
            return evaluate_batch(np.stack(cs))

        fit, scores, metrics = run_one(base, popsize or 1)
    else:
        episodes, run_cpu = cpu_evaluator(morph, terrains, seconds, payload,
                                          variants, seed, task)
        if not episodes[0].buildable:
            return _unbuildable(morph, base, terrains, episodes[0].mass)

        def run_one(c, n: int = 1):
            return run_cpu(c)

        def run_many(cs):
            return map_many(cs) if map_many else [run_one(c) for c in cs]

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
        cs = [full(np.asarray(z)) for z in candidates]
        for c, (fit, scores, metrics) in zip(cs, run_many(cs)):
            evals += 1
            losses.append(-fit)
            if fit > best.fitness:
                best = ControllerResult(c, fit, scores, metrics, evals)
        es.tell(candidates, losses)

    best.evals = evals
    if device == "warp":
        # The GPU only searched; the archive entry is scored on the CPU.
        best.fitness, best.scores, best.metrics = evaluate(
            [b.episode for b in batches], best.ctrl, var_sets, task)
    best.behaviour = behaviour_vector(morph, _nominal_flat(
        morph, terrains, best, variants, seconds, device, payload), task)
    return best


def _nominal_flat(morph, terrains, best, variants, seconds, device,
                  payload=0.0) -> Metrics:
    """Metrics of the best controller on flat ground, no randomised variant."""
    if variants == 0:
        for t, m in zip(terrains, best.metrics):
            if t.name == "flat":
                return m
    if device == "gpu":
        from darwinbot.sim_gpu import BatchEpisode
        return BatchEpisode(morph, flat(), seconds, payload).run_batch(best.ctrl[None])[0]
    # cpu and warp: the same CPU episode the archive is scored on.
    return Episode(morph, flat(), seconds, payload).run(best.ctrl)

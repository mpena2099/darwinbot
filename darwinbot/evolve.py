"""
evolve.py
---------
Outer loop: MAP-Elites over body plans.

The archive is a pyribs GridArchive indexed by morphology features (leg pairs,
wheeled pairs, number of servos). Each cell keeps the best body found with those
features, together with the controller CMA-ES learned for it. Instead of
converging on one design, the run ends with a map of the best walker, the best
wheeler, the best long-legged hexapod, and so on.

New bodies are mutated copies of random elites; the child starts its inner
optimisation from the parent's controller (Lamarckian inheritance), which is
much cheaper than learning a gait from scratch.
"""

import multiprocessing
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from ribs.archives import GridArchive

from darwinbot import controller as ctrl
from darwinbot import morphology as morpho
from darwinbot.inner import optimise_controller
from darwinbot.terrain import get_terrains

BODY_MODEL = 2                      # bump when the genome meaning changes
SERVO_MIN  = 4                      # one pair, no yaw, no wheel
SERVO_STEP = 2                      # both legs of a pair always add servos
N_SERVO_BINS = 11                   # 4, 6, ..., 24
SERVO_RANGE = (SERVO_MIN - 1, SERVO_MIN - 1 + SERVO_STEP * N_SERVO_BINS)

Cell = Tuple[int, int, int]


def cell_of(morph: morpho.Morphology) -> Cell:
    pairs, wheeled, servos = morpho.descriptor(morph)
    servo_bin = (servos - SERVO_RANGE[0]) // SERVO_STEP
    return (pairs, wheeled, int(np.clip(servo_bin, 0, N_SERVO_BINS - 1)))


def cell_name(cell: Cell) -> str:
    """Readable cell label, e.g. (3, 1, 5) -> '6legs-2wheels-14servos'."""
    pairs, wheeled, servo_bin = cell
    return (f"{2 * pairs}legs-{2 * wheeled}wheels-"
            f"{SERVO_MIN + SERVO_STEP * servo_bin}servos")


def parse_cell_name(name: str) -> Cell:
    try:
        legs, wheels, servos = name.strip().lower().split("-")
        servo_bin, rest = divmod(int(servos.removesuffix("servos")) - SERVO_MIN,
                                 SERVO_STEP)
        if rest or not 0 <= servo_bin < N_SERVO_BINS:
            raise ValueError
        return (int(legs.removesuffix("legs")) // 2,
                int(wheels.removesuffix("wheels")) // 2, servo_bin)
    except ValueError:
        raise ValueError(f"bad robot type '{name}', expected e.g. 6legs-2wheels-14servos")


@dataclass
class Elite:
    morph:   np.ndarray
    ctrl:    np.ndarray
    fitness: float
    scores:  List[float]
    born:    int          # batch index


@dataclass
class Archive:
    terrains: str
    history:  List[dict] = field(default_factory=list)
    batch:    int = 0
    evals:    int = 0
    seconds:  float = 8.0      # simulated seconds per episode in this run
    variants: int = 0          # randomised models per terrain in this run
    body_model: int = BODY_MODEL
    _grid:    Optional[GridArchive] = field(default=None, repr=False)

    def _ensure(self, n_scores: int) -> GridArchive:
        """Build the ribs archive on first insert, when score count is known."""
        if self._grid is None:
            self._grid = GridArchive(
                solution_dim=morpho.MORPH_LENGTH,
                dims=(morpho.N_PAIRS, morpho.N_PAIRS + 1, N_SERVO_BINS),
                ranges=((0.5, 3.5), (-0.5, 3.5), SERVO_RANGE),
                extra_fields={
                    "ctrl": ((ctrl.CTRL_LENGTH,), np.float64),
                    "scores": ((n_scores,), np.float64),
                    "born": ((), np.int64),
                },
            )
        return self._grid

    @property
    def cells(self) -> Dict[Cell, Elite]:
        """Read-only dict of the ribs archive, keyed by morphology cell."""
        if self._grid is None:
            return {}
        data = self._grid.data()
        cells = {}
        for i, genes in enumerate(data["solution"]):
            cells[cell_of(morpho.decode(genes))] = Elite(
                genes, data["ctrl"][i], float(data["objective"][i]),
                data["scores"][i].tolist(), int(data["born"][i]))
        return cells

    def insert(self, elite: Elite) -> bool:
        grid = self._ensure(len(elite.scores))
        measures = morpho.descriptor(morpho.decode(elite.morph))
        info = grid.add(solution=[elite.morph], objective=[elite.fitness],
                        measures=[measures], ctrl=[elite.ctrl],
                        scores=[elite.scores], born=[elite.born])
        return int(info["status"][0]) != 0

    def sample_elites(self, n: int, rng: np.random.Generator) -> dict:
        """Uniform over filled cells, drawing from the run's rng."""
        data = self._grid.data()
        idx = rng.integers(len(data["solution"]), size=n)
        return {k: v[idx] for k, v in data.items()}

    def __setstate__(self, state):
        # Archives pickled before the pyribs switch stored a plain `cells`
        # dict; replay them into a grid so old checkpoints still load.
        old_cells = state.pop("cells", None)
        grid = state.get("_grid")
        stored = grid.solution_dim if grid is not None else morpho.MORPH_LENGTH
        for elite in (old_cells or {}).values():
            stored = len(elite.morph)
        if stored != morpho.MORPH_LENGTH or state.get("body_model") != BODY_MODEL:
            raise ValueError("archive from an older body model, retrain")
        self.__dict__.update(state)
        self.__dict__.setdefault("_grid", None)
        self.__dict__.setdefault("seconds", 8.0)   # old runs used 8 s episodes
        self.__dict__.setdefault("variants", 0)    # and no randomisation
        for elite in (old_cells or {}).values():
            self.insert(elite)

    def best(self) -> Elite:
        return max(self.cells.values(), key=lambda e: e.fitness)

    def stats(self) -> dict:
        fits = [e.fitness for e in self.cells.values()]
        return {"batch": self.batch, "evals": self.evals, "cells": len(fits),
                "best": max(fits), "mean": float(np.mean(fits)),
                # QD-score: sum of fitness shifted to be positive per cell
                "qd_score": float(sum(f + 5.0 for f in fits))}


def _work(task):
    """Pool worker: optimise one body's controller. Runs in a subprocess."""
    (morph_genes, x0, terrains, budget, sigma0, seed, seconds,
     device, popsize, variants) = task
    morph = morpho.decode(morph_genes)
    res = optimise_controller(morph, get_terrains(terrains), x0=x0,
                              budget=budget, sigma0=sigma0, seed=seed,
                              seconds=seconds, device=device, popsize=popsize,
                              variants=variants)
    return morph_genes, res.ctrl, res.fitness, res.scores, res.evals


def run(
    terrains: str = "flat",
    n_init: int = 24,
    n_batches: int = 30,
    batch_size: int = 16,
    budget_init: int = 400,
    budget_child: int = 120,
    episode_seconds: float = 5.0,
    random_frac: float = 0.1,
    workers: Optional[int] = None,
    seed: int = 0,
    out_dir: str = "checkpoints",
    resume: Optional[str] = None,
    device: str = "cpu",
    popsize: Optional[int] = None,
    variants: int = 2,
) -> Archive:
    if device == "gpu" and variants > 0 and not resume:
        raise ValueError("randomisation is CPU only; use --variants 0 with --device gpu")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    if device == "gpu":
        print("[darwinbot] gpu: workers ignored, tasks run one at a time")
        workers = 1
        popsize = 64 if popsize is None else popsize
    else:
        workers = workers or max(1, multiprocessing.cpu_count() - 1)

    if resume:
        with open(resume, "rb") as f:
            archive: Archive = pickle.load(f)
        terrains = archive.terrains
        rng = np.random.default_rng(seed + archive.batch)
    else:
        archive = Archive(terrains=terrains, seconds=episode_seconds,
                          variants=variants)
    if device == "gpu" and archive.variants > 0:
        raise ValueError("randomisation is CPU only; use --variants 0 with --device gpu")

    def new_seed():
        return int(rng.integers(0, 2**31 - 1))

    def make_batch() -> list:
        if not archive.cells:
            return [(morpho.random_morphology(rng), None, terrains,
                     budget_init, 0.25, new_seed(), archive.seconds,
                     device, popsize, archive.variants) for _ in range(n_init)]
        parents = archive.sample_elites(batch_size, rng)
        tasks = []
        for i in range(batch_size):
            if rng.random() < random_frac:
                tasks.append((morpho.random_morphology(rng), None, terrains,
                              budget_init, 0.25, new_seed(), archive.seconds,
                              device, popsize, archive.variants))
            else:
                child = morpho.mutate(parents["solution"][i], rng)
                tasks.append((child, parents["ctrl"][i], terrains,
                              budget_child, 0.12, new_seed(), archive.seconds,
                              device, popsize, archive.variants))
        return tasks

    print(f"[darwinbot] terrains={terrains} workers={workers} device={device} "
          f"batches={n_batches} batch_size={batch_size}")
    pool = None if device == "gpu" else multiprocessing.Pool(workers)
    for _ in range(n_batches + (0 if archive.cells else 1)):
        t0 = time.time()
        added = 0
        tasks = make_batch()
        results = [_work(t) for t in tasks] if pool is None else pool.imap_unordered(_work, tasks)
        for genes, ctrl, fit, scores, evals in results:
            archive.evals += evals
            added += archive.insert(Elite(genes, ctrl, fit, scores, archive.batch))
        s = archive.stats()
        s.update(added=added, seconds=round(time.time() - t0, 1))
        archive.history.append(s)
        print(f"batch {s['batch']:3d}  cells {s['cells']:3d}  +{added:2d}  "
              f"best {s['best']:6.2f}  mean {s['mean']:6.2f}  "
              f"evals {s['evals']}  {s['seconds']}s", flush=True)
        archive.batch += 1
        with open(os.path.join(out_dir, "archive.pkl"), "wb") as f:
            pickle.dump(archive, f)
    if pool is not None:
        pool.close()
        pool.join()
    return archive

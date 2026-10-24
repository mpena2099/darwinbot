"""
evolve.py
---------
Outer loop: MAP-Elites over body plans.

The archive is a grid indexed by morphology features (leg pairs, wheeled
pairs, leg length). Each cell keeps the best body found with those features,
together with the controller CMA-ES learned for it. Instead of converging on
one design, the run ends with a map of the best walker, the best wheeler,
the best long-legged hexapod, and so on.

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

from darwinbot import morphology as morpho
from darwinbot.inner import optimise_controller
from darwinbot.terrain import get_terrains

LEG_LENGTH_BINS = np.linspace(0.10, 0.50, 6)   # 5 bins over thigh + shin

Cell = Tuple[int, int, int]


def cell_of(morph: morpho.Morphology) -> Cell:
    pairs, wheeled, leg = morpho.descriptor(morph)
    leg_bin = int(np.clip(np.digitize(leg, LEG_LENGTH_BINS) - 1, 0,
                          len(LEG_LENGTH_BINS) - 2))
    return (pairs, wheeled, leg_bin)


LEG_LENGTH_NAMES = ("xshort", "short", "medium", "long", "xlong")


def cell_name(cell: Cell) -> str:
    """Readable cell label, e.g. (3, 0, 2) -> '6legs-0wheels-medium'."""
    pairs, wheeled, leg_bin = cell
    return f"{2 * pairs}legs-{2 * wheeled}wheels-{LEG_LENGTH_NAMES[leg_bin]}"


def parse_cell_name(name: str) -> Cell:
    try:
        legs, wheels, length = name.strip().lower().split("-")
        return (int(legs.removesuffix("legs")) // 2,
                int(wheels.removesuffix("wheels")) // 2,
                LEG_LENGTH_NAMES.index(length))
    except ValueError:
        raise ValueError(f"bad robot type '{name}', expected e.g. 6legs-0wheels-medium")


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
    cells:    Dict[Cell, Elite] = field(default_factory=dict)
    history:  List[dict] = field(default_factory=list)
    batch:    int = 0
    evals:    int = 0

    def insert(self, elite: Elite) -> bool:
        cell = cell_of(morpho.decode(elite.morph))
        old = self.cells.get(cell)
        if old is None or elite.fitness > old.fitness:
            self.cells[cell] = elite
            return True
        return False

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
    morph_genes, x0, terrains, budget, sigma0, seed = task
    morph = morpho.decode(morph_genes)
    res = optimise_controller(morph, get_terrains(terrains), x0=x0,
                              budget=budget, sigma0=sigma0, seed=seed)
    return morph_genes, res.ctrl, res.fitness, res.scores, res.evals


def run(
    terrains: str = "flat",
    n_init: int = 24,
    n_batches: int = 30,
    batch_size: int = 16,
    budget_init: int = 400,
    budget_child: int = 200,
    random_frac: float = 0.1,
    workers: Optional[int] = None,
    seed: int = 0,
    out_dir: str = "checkpoints",
    resume: Optional[str] = None,
) -> Archive:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    workers = workers or max(1, multiprocessing.cpu_count() - 1)

    if resume:
        with open(resume, "rb") as f:
            archive: Archive = pickle.load(f)
        terrains = archive.terrains
        rng = np.random.default_rng(seed + archive.batch)
    else:
        archive = Archive(terrains=terrains)

    def new_seed():
        return int(rng.integers(0, 2**31 - 1))

    def make_batch() -> list:
        if not archive.cells:
            return [(morpho.random_morphology(rng), None, terrains,
                     budget_init, 0.25, new_seed()) for _ in range(n_init)]
        elites = list(archive.cells.values())
        tasks = []
        for _ in range(batch_size):
            if rng.random() < random_frac:
                tasks.append((morpho.random_morphology(rng), None, terrains,
                              budget_init, 0.25, new_seed()))
            else:
                parent = elites[rng.integers(len(elites))]
                child = morpho.mutate(parent.morph, rng)
                tasks.append((child, parent.ctrl, terrains,
                              budget_child, 0.12, new_seed()))
        return tasks

    print(f"[darwinbot] terrains={terrains} workers={workers} "
          f"batches={n_batches} batch_size={batch_size}")
    with multiprocessing.Pool(workers) as pool:
        for _ in range(n_batches + (0 if archive.cells else 1)):
            t0 = time.time()
            added = 0
            for genes, ctrl, fit, scores, evals in pool.imap_unordered(_work, make_batch()):
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
    return archive

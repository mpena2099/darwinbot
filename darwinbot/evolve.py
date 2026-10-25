"""
evolve.py
---------
Outer loop: quality diversity over body plans.

The archive is a pyribs CVTArchive (about 500 cells) over four behaviour and
cost descriptors, all measured on the nominal flat-terrain episode of the
body's best controller: duty (fraction of time the terminal points touch the
ground), height (mean torso height over the body's reach), cost (number of
servos) and cost of transport (work per weight and distance, log scale), which
lets an efficient walker keep its cell next to a faster but hungrier roller. Each cell keeps the best body found there, together with the
controller CMA-ES learned for it. Instead of converging on one design, the run
ends with a map of the fastest crawler, the fastest roller, the best tall
trotter, and so on.

New bodies are mutated copies of elites drawn with probability proportional to
their novelty (mean distance to the k nearest elites in descriptor space), or
uniformly with parent_selection="uniform". The child starts its inner
optimisation from the parent's controller (Lamarckian inheritance), which is
much cheaper than learning a gait from scratch.
"""

import hashlib
import multiprocessing
import os
import pickle
import re
import signal
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from ribs.archives import CVTArchive

from darwinbot import controller as ctrl
from darwinbot import morphology as morpho
from darwinbot.inner import optimise_controller
from darwinbot.sim import COT_MAX, COT_MIN, HEIGHT_MAX
from darwinbot.terrain import get_terrains

BODY_MODEL = 8                      # bump when the genome meaning changes
N_CELLS    = 500                    # CVT cells
CVT_SEED   = 0                      # fixed seed: same centroids in every run
K_NOVELTY  = 10                     # neighbours in the novelty score
PARENT_SELECTIONS = ("novelty", "uniform")

# Name bins. Height is torso z over body reach, duty is the contact fraction.
HEIGHT_BINS = ((0.4, "low"), (0.8, "mid"), (float("inf"), "high"))
ROLL_SHARE  = 0.5                   # wheel share of contact time above this: roll
CRAWL_DUTY, WALK_DUTY, BELLY_DUTY = 0.8, 0.5, 0.15

Cell = int                          # index of the CVT cell


N_DESCRIPTORS = 4


def normalise(behaviour) -> np.ndarray:
    """Descriptor in [0, 1]^4 from a behaviour vector, so that every axis
    weighs the same. Cost of transport spans decades, so it is taken in log."""
    duty, height, servos, _, cot = np.asarray(behaviour, dtype=float)[:5]
    cot = np.log(np.clip(cot, COT_MIN, COT_MAX) / COT_MIN) / np.log(COT_MAX / COT_MIN)
    return np.clip([duty, height / HEIGHT_MAX, servos / morpho.MAX_SERVOS, cot], 0.0, 1.0)


@lru_cache(maxsize=1)
def _centroids() -> np.ndarray:
    probe = CVTArchive(solution_dim=1, centroids=N_CELLS, ranges=[(0.0, 1.0)] * N_DESCRIPTORS,
                       seed=CVT_SEED, samples=20000)
    return np.array(probe.centroids)


def cell_of(behaviour) -> Cell:
    """CVT cell of a behaviour vector (duty, height, servos, wheel share, cot)."""
    point = normalise(behaviour)
    return int(np.argmin(((_centroids() - point) ** 2).sum(axis=1)))


def type_name(behaviour) -> str:
    """Readable type from a behaviour vector, e.g. '8servos-low-crawl'."""
    duty, height, servos, wheel_share = (float(v) for v in behaviour[:4])
    band = next(name for top, name in HEIGHT_BINS if height < top)
    if wheel_share > ROLL_SHARE:
        gait = "roll"
    elif duty > CRAWL_DUTY:
        gait = "crawl"
    elif duty >= WALK_DUTY:
        gait = "walk"
    elif duty >= BELLY_DUTY:
        gait = "trot"
    else:
        # tips barely touch the ground: the robot slides on its body
        gait = "belly"
    return f"{int(round(servos))}servos-{band}-{gait}"


_NAME = re.compile(r"^(\d+)servos-(low|mid|high)-(crawl|walk|trot|belly|roll)(?:-(\d+))?$")


def parse_cell_name(name: str) -> Tuple[str, Optional[int]]:
    """(type name, cell index or None) of a name printed by Archive.names()."""
    m = _NAME.match(name.strip().lower())
    if not m:
        raise ValueError(f"bad robot type '{name}', expected e.g. 8servos-low-crawl")
    base = f"{m.group(1)}servos-{m.group(2)}-{m.group(3)}"
    return base, int(m.group(4)) if m.group(4) else None


@dataclass
class Elite:
    morph:   np.ndarray
    ctrl:    np.ndarray
    fitness: float
    scores:  List[float]
    born:    int          # batch index
    # (duty, height, servos, wheel share, cot); None: a body never simulated
    behaviour: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.behaviour is None:
            self.behaviour = np.array(
                [0.0, 0.0, float(morpho.decode(self.morph).n_servos), 0.0, COT_MAX])
        self.behaviour = np.asarray(self.behaviour, dtype=float)


def elite_id(morph_genes, ctrl_genes) -> str:
    """Stable id of an elite: hash of its body and controller genes, so it can
    be recomputed from any copy of the elite (the CMA-ES result never repeats)."""
    h = hashlib.blake2b(np.asarray(morph_genes, np.float64).tobytes(), digest_size=8)
    h.update(np.asarray(ctrl_genes, np.float64).tobytes())
    return h.hexdigest()


@dataclass
class Archive:
    terrains: str
    history:  List[dict] = field(default_factory=list)
    # id -> (parent id or None, batch, fitness, type name) of every elite that
    # ever entered the archive, so a lineage survives its ancestors' eviction
    lineage:  Dict[str, tuple] = field(default_factory=dict)
    batch:    int = 0
    evals:    int = 0
    seconds:  float = 8.0      # simulated seconds per episode in this run
    variants: int = 0          # randomised models per terrain in this run
    body_model: int = BODY_MODEL
    _grid:    Optional[CVTArchive] = field(default=None, repr=False)

    def _ensure(self, n_scores: int) -> CVTArchive:
        """Build the ribs archive on first insert, when score count is known."""
        if self._grid is None:
            self._grid = CVTArchive(
                solution_dim=morpho.MORPH_LENGTH,
                centroids=_centroids(),
                ranges=[(0.0, 1.0)] * N_DESCRIPTORS,
                extra_fields={
                    "ctrl": ((ctrl.CTRL_LENGTH,), np.float64),
                    "scores": ((n_scores,), np.float64),
                    "born": ((), np.int64),
                    "behaviour": ((5,), np.float64),
                },
            )
        return self._grid

    @property
    def cells(self) -> Dict[Cell, Elite]:
        """Read-only dict of the ribs archive, keyed by CVT cell index."""
        if self._grid is None:
            return {}
        data = self._grid.data()
        return {int(data["index"][i]): Elite(
                    data["solution"][i], data["ctrl"][i], float(data["objective"][i]),
                    data["scores"][i].tolist(), int(data["born"][i]),
                    data["behaviour"][i])
                for i in range(len(data["solution"]))}

    def names(self) -> Dict[Cell, str]:
        """Unique readable name of every filled cell. Cells that share a type
        name get their cell index appended."""
        cells = self.cells
        base = {c: type_name(e.behaviour) for c, e in cells.items()}
        count: Dict[str, int] = {}
        for n in base.values():
            count[n] = count.get(n, 0) + 1
        return {c: n if count[n] == 1 else f"{n}-{c}" for c, n in base.items()}

    def cell_name(self, cell: Cell) -> str:
        return self.names()[cell]

    def find(self, name: str) -> Cell:
        """Cell of a name printed by names(); ValueError when absent or ambiguous."""
        base, index = parse_cell_name(name)
        names = self.names()
        if index is not None:
            if names.get(index, "").rsplit("-", 1)[0] != base:
                raise ValueError(f"no {name} robot in the archive")
            return index
        hits = [c for c, n in names.items() if n == base]
        if hits:
            return hits[0]
        if any(n.startswith(base + "-") for n in names.values()):
            raise ValueError(f"'{name}' is ambiguous, use one of: " + ", ".join(
                sorted(n for n in names.values() if n.startswith(base + "-"))))
        raise ValueError(f"no {name} robot in the archive")

    def insert(self, elite: Elite, parent: Optional[str] = None) -> bool:
        grid = self._ensure(len(elite.scores))
        info = grid.add(solution=[elite.morph], objective=[elite.fitness],
                        measures=[normalise(elite.behaviour)],
                        ctrl=[elite.ctrl], scores=[elite.scores], born=[elite.born],
                        behaviour=[elite.behaviour])
        added = int(info["status"][0]) != 0
        if added:
            self.lineage[elite_id(elite.morph, elite.ctrl)] = (
                parent, elite.born, float(elite.fitness), type_name(elite.behaviour))
        return added

    def ancestors(self, elite: Elite) -> List[Tuple[str, tuple]]:
        """(id, lineage record) from the elite back to its first ancestor; empty
        for archives saved before lineage was recorded."""
        out, uid = [], elite_id(elite.morph, elite.ctrl)
        while uid in self.lineage and len(out) <= len(self.lineage):
            out.append((uid, self.lineage[uid]))
            uid = self.lineage[uid][0]
        return out

    def novelty(self) -> np.ndarray:
        """Per elite (in data() order): mean distance to its K_NOVELTY nearest
        elites in the normalised descriptor space."""
        if self._grid is None:
            return np.zeros(0)
        pts = self._grid.data()["measures"]
        n = len(pts)
        if n < 2:
            return np.zeros(n)
        dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        k = min(K_NOVELTY, n - 1)
        return np.sort(dist, axis=1)[:, :k].mean(axis=1)

    def sample_elites(self, n: int, rng: np.random.Generator,
                      selection: str = "novelty") -> dict:
        """Parents drawn with the run's rng: probability proportional to novelty
        (falls back to uniform when every elite has zero novelty)."""
        data = self._grid.data()
        size = len(data["solution"])
        weights = self.novelty() if selection == "novelty" else np.zeros(size)
        p = weights / weights.sum() if weights.sum() > 0 else None
        idx = rng.choice(size, size=n, p=p)
        return {k: v[idx] for k, v in data.items()}

    def __setstate__(self, state):
        grid = state.get("_grid")
        if (state.get("body_model") != BODY_MODEL or "cells" in state
                or (grid is not None and not isinstance(grid, CVTArchive))):
            raise ValueError("archive from an older body model, retrain")
        self.__dict__.update(state)
        self.__dict__.setdefault("_grid", None)
        self.__dict__.setdefault("seconds", 8.0)   # old runs used 8 s episodes
        self.__dict__.setdefault("variants", 0)    # and no randomisation
        self.__dict__.setdefault("lineage", {})    # nor lineage records

    def best(self) -> Elite:
        return max(self.cells.values(), key=lambda e: e.fitness)

    def stats(self) -> dict:
        fits = [e.fitness for e in self.cells.values()]
        nov = self.novelty()
        return {"batch": self.batch, "evals": self.evals, "cells": len(fits),
                "best": max(fits), "mean": float(np.mean(fits)),
                "novelty": float(nov.mean()) if len(nov) else 0.0,
                # QD-score: sum of fitness shifted to be positive per cell
                "qd_score": float(sum(f + 5.0 for f in fits))}


def _ignore_sigint():
    """Pool workers leave Ctrl+C to the main process, which stops them."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _work(task):
    """Pool worker: optimise one body's controller. Runs in a subprocess."""
    (morph_genes, x0, terrains, budget, sigma0, seed, seconds,
     device, popsize, variants, parent) = task
    morph = morpho.decode(morph_genes)
    res = optimise_controller(morph, get_terrains(terrains), x0=x0,
                              budget=budget, sigma0=sigma0, seed=seed,
                              seconds=seconds, device=device, popsize=popsize,
                              variants=variants)
    return (morph_genes, res.ctrl, res.fitness, res.scores, res.evals,
            res.behaviour, parent)


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
    parent_selection: str = "novelty",
) -> Archive:
    if parent_selection not in PARENT_SELECTIONS:
        raise ValueError(f"parent_selection must be one of {PARENT_SELECTIONS}")
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
                     device, popsize, archive.variants, None) for _ in range(n_init)]
        parents = archive.sample_elites(batch_size, rng, parent_selection)
        tasks = []
        for i in range(batch_size):
            if rng.random() < random_frac:
                tasks.append((morpho.random_morphology(rng), None, terrains,
                              budget_init, 0.25, new_seed(), archive.seconds,
                              device, popsize, archive.variants, None))
            else:
                child = morpho.mutate(parents["solution"][i], rng)
                tasks.append((child, parents["ctrl"][i], terrains,
                              budget_child, 0.12, new_seed(), archive.seconds,
                              device, popsize, archive.variants,
                              elite_id(parents["solution"][i], parents["ctrl"][i])))
        return tasks

    print(f"[darwinbot] terrains={terrains} workers={workers} device={device} "
          f"batches={n_batches} batch_size={batch_size}")
    pool = None if device == "gpu" else multiprocessing.Pool(workers, _ignore_sigint)
    path = os.path.join(out_dir, "archive.pkl")
    saved = None                                  # batch count in the file on disk
    try:
        for _ in range(n_batches + (0 if archive.cells else 1)):
            t0 = time.time()
            added = 0
            tasks = make_batch()
            results = ([_work(t) for t in tasks] if pool is None
                       else pool.imap_unordered(_work, tasks))
            for genes, ctrl, fit, scores, evals, behaviour, parent in results:
                archive.evals += evals
                added += archive.insert(Elite(genes, ctrl, fit, scores,
                                              archive.batch, behaviour), parent)
            s = archive.stats()
            s.update(added=added, seconds=round(time.time() - t0, 1))
            archive.history.append(s)
            print(f"batch {s['batch']:3d}  cells {s['cells']:3d}  +{added:2d}  "
                  f"best {s['best']:6.2f}  mean {s['mean']:6.2f}  "
                  f"nov {s['novelty']:.3f}  evals {s['evals']}  {s['seconds']}s",
                  flush=True)
            archive.batch += 1
            with open(path + ".tmp", "wb") as f:
                pickle.dump(archive, f)
            os.replace(path + ".tmp", path)      # a Ctrl+C never leaves half a file
            saved = archive.batch
    except KeyboardInterrupt:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        # The file on disk is the archive saved after the last finished batch.
        if saved is None:
            print("stopped before the first batch finished, nothing saved")
        else:
            print(f"stopped, archive saved at batch {saved}")
        return archive
    if pool is not None:
        pool.close()
        pool.join()
    return archive

"""
evolve.py
---------
Outer loop: quality diversity over body plans.

The archive is a pyribs CVTArchive (about 800 cells) over five behaviour and
cost descriptors, all measured on the nominal flat-terrain episode of the
body's best controller (the servo count is a property of the body): number of
servos (the cost of building the robot, so a 6-servo body and a 16-servo one
that move alike never compete for a cell), duty (fraction of time the terminal points touch the
ground), height (mean torso height over the body's reach), number of legs
(limbs on the torso with a joint and no wheel, so a hexapod never competes with
a two-legged crawler for a cell) and cost of transport (work per weight and
distance, log scale), which lets an efficient walker keep its cell next to a
faster but hungrier roller. Each cell keeps the best body found there, together with the
controller CMA-ES learned for it. Instead of converging on one design, the run
ends with a map of the fastest crawler, the fastest roller, the best tall
trotter, and so on.

An experiment may add body_ground (the share of ticks the torso itself touches
the ground) as a sixth axis (`train.py --axes body`), which only applies to a
new archive: `Archive.axes` says which descriptors a grid holds. The release 1
axes (`train.py --axes r1`) keep the four original descriptors, without the
servo count, in a 500-cell archive.

New bodies are mutated copies of elites drawn with probability proportional to
their novelty (mean distance to the k nearest elites in descriptor space), or
uniformly with parent_selection="uniform". The child starts its inner
optimisation from the parent's controller (Lamarckian inheritance), which is
much cheaper than learning a gait from scratch.
"""

import contextlib
import copy
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import re
import signal
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from ribs.archives import CVTArchive

from darwinbot import ablate, body_options, fastloop, fitness, hardware, plugins, rules, trace
from darwinbot import controller as ctrl
from darwinbot import morphology as morpho
from darwinbot.fitness import commands, task_names, task_score
from darwinbot.inner import evaluate, optimise_controller
from darwinbot.sim import (COT_MAX, COT_MIN, EPISODE_SECONDS, HEIGHT_MAX, Behaviour,
                           Episode)
from darwinbot.terrain import flat, get_terrains

BODY_MODEL = 26                     # bump when the genome meaning changes
N_CELLS    = 800                    # CVT cells of the default and body archives
CVT_SEED   = 0                      # fixed seed: same centroids in every run
K_NOVELTY  = 10                     # neighbours in the novelty score
MAX_SEED_TRIES = 20                 # random batches tried to fill an empty archive
PARENT_SELECTIONS = ("novelty", "uniform")
INNER_STRATEGIES = (None, "diagonal", "focused", "diagonal+focused")   # inner.optimise_controller
VARIANT_MODES = ("all", "first")

# from-version -> function translating one elite's (morph, ctrl) genes to the
# next version. Identity when only the physics or the printed parts changed. A
# version with no entry cannot be migrated: its archives are refused. A genome
# change adds its entry: decode with the old constants (kept next to the
# function as a small frozen dict) and encode with the new ones; new genes get
# the neutral value.
# Frozen constants of the versions a translation reads. Not the old code.
_V20 = {"wheel_range": (0.03, 0.06),          # m, wheel radius gene range
        "ctrl_per_node": 8}                    # amp1 .. steer_shift
_V21 = {"wheel_range": (0.03, 0.10),
        "wheel_width": lambda rad: max(0.02, 0.5 * rad),   # m: the wheel was
                                               # 2 * max(10 mm, rad / 4) wide, no gene
        "ctrl_per_node": 10}                   # amp1 .. speed_shift
_V23 = {"ctrl_per_node": 10}                   # amp1 .. speed_shift


def _wheel_nodes(genes) -> List[int]:
    """Slots that decode to a wheel (the kind does not depend on the ranges)."""
    return [i for i, n in morpho.decode(genes).nodes.items() if n.kind == "wheel"]


def _migrate_20(genes, cvec):
    """20 -> 21: the wheel radius range grew from 60 to 100 mm; each wheel gene
    is rescaled so its radius in metres stays."""
    genes = np.array(genes, dtype=float)
    blocks = genes.reshape(morpho.MAX_NODES, morpho.BLOCK_GENES)
    for i in _wheel_nodes(genes):
        rad = morpho._scale(blocks[i, morpho.WHEEL], *_V20["wheel_range"])
        blocks[i, morpho.WHEEL] = morpho._unscale(rad, *_V21["wheel_range"])
    return genes, cvec


def _migrate_21(genes, cvec):
    """21 -> 22: a wheel's width became the WIDTH gene (the old code derived it
    from the radius: the gene is set to that width), and every controller slot
    gained speed_gain and speed_shift (neutral: 0.5 is a gain and a shift of 0,
    the same gait at every speed command)."""
    genes = np.array(genes, dtype=float)
    blocks = genes.reshape(morpho.MAX_NODES, morpho.BLOCK_GENES)
    for i in _wheel_nodes(genes):
        rad = morpho._scale(blocks[i, morpho.WHEEL], *_V21["wheel_range"])
        blocks[i, morpho.WIDTH] = morpho._unscale(_V21["wheel_width"](rad),
                                                  *morpho.WHEEL_WIDTH_RANGE)
    cvec = np.asarray(cvec, dtype=float)
    per = _V20["ctrl_per_node"]
    if len(cvec) == 1 + morpho.MAX_NODES * per:
        slots = cvec[1:].reshape(morpho.MAX_NODES, per)
        neutral = np.full((morpho.MAX_NODES, _V21["ctrl_per_node"] - per), 0.5)
        cvec = np.concatenate([cvec[:1], np.hstack([slots, neutral]).reshape(-1)])
    return genes, cvec


def _migrate_23(genes, cvec):
    """23 -> 24: every controller slot gained couple_k and couple_lag, set to
    the neutral k = 0 and lag = 0 rad, so the old controller is unchanged."""
    cvec = np.asarray(cvec, dtype=float)
    per = _V23["ctrl_per_node"]
    if len(cvec) == 1 + morpho.MAX_NODES * per:
        slots = cvec[1:].reshape(morpho.MAX_NODES, per)
        extra = np.tile([0.0, 0.5], (morpho.MAX_NODES, 1))
        cvec = np.concatenate([cvec[:1], np.hstack([slots, extra]).reshape(-1)])
    return genes, cvec


MIGRATIONS = {20: _migrate_20, 21: _migrate_21,
              22: lambda morph, cvec: (morph, cvec), 23: _migrate_23,
              24: lambda morph, cvec: (morph, cvec),   # 24: the elbow fin, parts only
              25: lambda morph, cvec: (morph, cvec)}   # 25: the fall tick counts, physics only
_stale_ok = False                   # set by loading_stale()


def migration_path(version) -> Optional[List[int]]:
    """Versions whose MIGRATIONS entries take an archive from `version` to
    BODY_MODEL, or None when one is missing (or the archive is newer)."""
    if not isinstance(version, int) or version > BODY_MODEL:
        return None
    path = list(range(version, BODY_MODEL))
    return path if all(v in MIGRATIONS for v in path) else None


@contextlib.contextmanager
def loading_stale():
    """Let pickle.load return an archive of an older body model. Such an
    archive carries its old body_model and must go through
    experiments/migrate_archive.py before it is trained, scored or replayed."""
    global _stale_ok
    _stale_ok = True
    try:
        yield
    finally:
        _stale_ok = False

def load_archive(path, stale: bool = False):
    """Open an archive pickle. A missing file or an archive of another body model
    stops with a plain message; stale=True returns an archive of an older body
    model instead (for experiments/migrate_archive.py)."""
    try:
        with open(path, "rb") as f:
            if stale:
                with loading_stale():
                    archive = pickle.load(f)
            else:
                archive = pickle.load(f)
        archive.assign_uids()
        return archive
    except FileNotFoundError:
        raise SystemExit(f"{path} not found: train first with experiments/train.py")
    except ValueError as e:
        raise SystemExit(f"{path}: {e}. Train a new archive with experiments/train.py "
                         f"(or pass --archive with a newer one).")


def score_options(basal_scale=None, turn_scale=None, turn_in_place=None,
                  turn_cap_deg=None) -> dict:
    """Non-default score settings from train.py flags and the environment."""
    return {**rules.score_option(basal_scale),
            **fitness.score_options(turn_scale, turn_in_place, turn_cap_deg)}


def apply_score(score: dict) -> None:
    """Apply an archive's score settings before scoring (env wins while set)."""
    rules.apply_score(score)
    fitness.apply_score(score)


# Name bins. Height is torso z over body reach, duty is the contact fraction.
HEIGHT_BINS = ((0.4, "low"), (0.8, "mid"), (float("inf"), "high"))
ROLL_SHARE  = 0.5                   # wheel share of contact time above this: roll
BODY_SHARE  = 0.5                   # body (not a terminal point) share of ticks above this: belly
CRAWL_DUTY, WALK_DUTY, BELLY_DUTY = 0.8, 0.5, 0.15

Cell = int                          # index of the CVT cell


N_DESCRIPTORS = 5
R1_DESCRIPTORS = 4              # r1: duty, height, legs, cot (no servo count)
R1_CELLS = 500                  # r1 CVT grid, cached apart from the 800-cell grids
AXES = ("default", "body", "r1")   # experiment options, stored in the Archive
AXES_SPEC = {                   # axes -> (descriptors, CVT cells)
    "default": (N_DESCRIPTORS, N_CELLS),
    "body": (N_DESCRIPTORS + 1, N_CELLS),
    "r1": (R1_DESCRIPTORS, R1_CELLS),
}


def _descriptor_count(axes_or_n) -> int:
    """Axes name or descriptor count -> descriptor count (call sites that hold
    an Archive pass n_descriptors; scripts may pass the name instead)."""
    return AXES_SPEC[axes_or_n][0] if isinstance(axes_or_n, str) else int(axes_or_n)


def normalise(behaviour, n_descriptors: int = N_DESCRIPTORS) -> np.ndarray:
    """Descriptor in [0, 1]^n_descriptors (duty, height, servos, legs, cot and,
    for a body archive, body_ground: the archive axes, not the order of the
    Behaviour fields) from a Behaviour, so that every axis weighs the same. Cost
    of transport spans decades, so it is taken in log. The r1 archive drops the
    servo-count axis."""
    n = _descriptor_count(n_descriptors)
    b = Behaviour.of(behaviour)
    cot = np.log(np.clip(b.cot, COT_MIN, COT_MAX) / COT_MIN) / np.log(COT_MAX / COT_MIN)
    if n == R1_DESCRIPTORS:
        point = [b.duty, b.height / HEIGHT_MAX, b.legs / morpho.MAX_LEGS, cot]
    else:
        point = [b.duty, b.height / HEIGHT_MAX, b.servos / morpho.MAX_SERVOS,
                 b.legs / morpho.MAX_LEGS, cot]
        if n > N_DESCRIPTORS:
            point.append(b.body_ground)     # already a share
    return np.clip(point[:n], 0.0, 1.0)


@lru_cache(maxsize=None)             # one centroid set per axis count
def _centroids(n_descriptors: int = N_DESCRIPTORS) -> np.ndarray:
    n = _descriptor_count(n_descriptors)
    probe = CVTArchive(solution_dim=1,
                       centroids=R1_CELLS if n == R1_DESCRIPTORS else N_CELLS,
                       ranges=[(0.0, 1.0)] * n,
                       seed=CVT_SEED, samples=20000)
    return np.array(probe.centroids)


def cell_of(behaviour, n_descriptors: int = N_DESCRIPTORS) -> Cell:
    """CVT cell of a Behaviour in the given archive axes."""
    n = _descriptor_count(n_descriptors)
    point = normalise(behaviour, n)
    return int(np.argmin(((_centroids(n) - point) ** 2).sum(axis=1)))


def type_name(behaviour) -> str:
    """Readable type from a Behaviour, e.g. '8servos-low-crawl'."""
    b = Behaviour.of(behaviour)
    band = next(name for top, name in HEIGHT_BINS if b.height < top)
    if b.body_ground > BODY_SHARE:
        # the body itself drags on the ground, wheels or not: it slides on its belly
        gait = "belly"
    elif b.wheel_share > ROLL_SHARE:
        gait = "roll"
    elif b.duty > CRAWL_DUTY:
        gait = "crawl"
    elif b.duty >= WALK_DUTY:
        gait = "walk"
    elif b.duty >= BELLY_DUTY:
        gait = "trot"
    else:
        # tips barely touch the ground: the robot slides on its body
        gait = "belly"
    return f"{int(round(b.servos))}servos-{band}-{gait}"


_NAME = re.compile(r"^(?:(\d+)servos-(low|mid|high)-(crawl|walk|trot|belly|roll))?"
                   r"(?:-?(\d+))?$")


def parse_cell_name(name: str) -> Tuple[Optional[str], Optional[int]]:
    """(type name or None, uid or None) of a name printed by Archive.names():
    '8servos-low-crawl-17', the bare uid '17' or the bare type '8servos-low-crawl'."""
    text = name.strip().lower()
    m = _NAME.match(text) if text else None
    if not m or not (m.group(1) or m.group(4)):
        raise ValueError(f"bad robot name '{name}', expected <type>-<uid> "
                         "(e.g. 8servos-low-crawl-17), a uid (17) or a type "
                         "(8servos-low-crawl)")
    base = f"{m.group(1)}servos-{m.group(2)}-{m.group(3)}" if m.group(1) else None
    return base, int(m.group(4)) if m.group(4) else None


@dataclass
class Elite:
    morph:   np.ndarray
    ctrl:    np.ndarray
    fitness: float
    scores:  List[float]
    born:    int          # batch index
    # None: a body never simulated
    behaviour: Optional[Behaviour] = None
    # stable id of the robot inside its archive; -1 until an archive numbers it.
    # Copies made with dataclasses.replace (polish, prune, migrate) keep it.
    uid: int = -1

    def __post_init__(self):
        if self.behaviour is None:
            m = morpho.decode(self.morph)
            self.behaviour = Behaviour(0.0, 0.0, float(m.n_servos), 0.0, COT_MAX,
                                       float(m.n_legs))
        self.behaviour = Behaviour.of(self.behaviour)

    def __setstate__(self, state):
        # elites pickled before Behaviour carry a plain array or tuple
        self.__dict__.update(state)
        self.__dict__.setdefault("uid", -1)
        if self.__dict__.get("behaviour") is not None:
            self.behaviour = Behaviour.of(self.behaviour)


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
    task:     str = "forward"  # fitness.TASKS: how episodes are scored
    payload:  float = 0.0      # extra torso mass in kg
    ablate:   tuple = ()       # ablate.NAMES switched off in this run (experiments)
    rules:    tuple = ()       # rules.NAMES switched on in this run (experiments)
    score:    dict = field(default_factory=dict)   # non-default score settings, from train.py
    body:     str = ""         # body_options limits, as given to train.py --body
    axes:     str = "default"  # AXES: default, body or r1 (new archives only)
    profile:  dict = field(default_factory=lambda: copy.deepcopy(hardware.PROFILE))
    locked:   tuple = ()       # node ids mutation must keep (0: the torso), from seed.py
    plugin:   dict = field(default_factory=dict)   # name and sha256 of the --plugin file
    # search options of the run that wrote it (not the score): SEARCH_FLAGS keys
    search:   dict = field(default_factory=dict)
    body_model: int = BODY_MODEL
    uids:     Dict[Cell, int] = field(default_factory=dict)   # cell -> uid of its elite
    next_uid: int = 0          # the next uid to hand out; never reused
    _grid:    Optional[CVTArchive] = field(default=None, repr=False)

    @property
    def n_descriptors(self) -> int:
        """Archive axes: five by default, six with body_ground, four for r1."""
        return AXES_SPEC.get(self.axes, AXES_SPEC["default"])[0]

    def _ensure(self, n_scores: int) -> CVTArchive:
        """Build the ribs archive on first insert, when score count is known."""
        if self._grid is None:
            self._grid = CVTArchive(
                solution_dim=morpho.MORPH_LENGTH,
                centroids=_centroids(self.n_descriptors),
                ranges=[(0.0, 1.0)] * self.n_descriptors,
                extra_fields={
                    "ctrl": ((ctrl.CTRL_LENGTH,), np.float64),
                    "scores": ((n_scores,), np.float64),
                    "born": ((), np.int64),
                    "behaviour": ((len(Behaviour._fields),), np.float64),
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
                    data["behaviour"][i], self.uids.get(int(data["index"][i]), -1))
                for i in range(len(data["solution"]))}

    def like(self) -> "Archive":
        """Empty archive with this run's settings, but no cells, history or lineage."""
        return Archive(terrains=self.terrains, batch=self.batch, evals=self.evals,
                       seconds=self.seconds, variants=self.variants, task=self.task,
                       payload=self.payload, ablate=self.ablate, rules=self.rules,
                       score=dict(self.score),
                       body=self.body, axes=self.axes, profile=copy.deepcopy(self.profile),
                       locked=self.locked, plugin=dict(self.plugin),
                       next_uid=self.next_uid)

    def activate(self) -> None:
        """Turn on the ablations, score rules, score settings and body limits
        this archive was trained with; scripts that score, decode or build its
        robots call this."""
        ablate.set_active(self.ablate)
        rules.set_active(self.rules)
        body_options.set_active(self.body)
        apply_score(self.score)

    def assign_uids(self) -> None:
        """Number the cells that have no uid (archives saved before uids), in
        cell order, and change nothing else."""
        for c in sorted(self.cells):
            if c not in self.uids:
                self.uids[c] = self.next_uid
                self.next_uid += 1

    def names(self) -> Dict[Cell, str]:
        """Name of every filled cell: '<type>-<uid>'. The uid never changes, so
        a robot keeps its name while others come and go."""
        return {c: f"{type_name(e.behaviour)}-{e.uid}" for c, e in self.cells.items()}

    def find(self, name: str) -> Cell:
        """Cell of a robot by full name ('8servos-low-crawl-17'), bare uid ('17')
        or a bare type that only one robot has. A uid whose type part is stale
        (the body was pruned) still finds the robot and prints its current name.
        ValueError when absent or ambiguous."""
        base, uid = parse_cell_name(name)
        cells = self.cells
        if uid is not None:
            hits = [c for c, e in cells.items() if e.uid == uid]
            if not hits:
                raise ValueError(f"no robot with uid {uid} in the archive (names are "
                                 "<type>-<uid>, e.g. 8servos-low-crawl-17)")
            current = type_name(cells[hits[0]].behaviour)
            if base is not None and base != current:
                print(f"note: {name} is now {current}-{uid}", flush=True)
            return hits[0]
        hits = [c for c, e in cells.items() if type_name(e.behaviour) == base]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise ValueError(f"'{name}' is ambiguous, use one of: " + ", ".join(
                sorted(f"{base}-{cells[c].uid}" for c in hits)))
        raise ValueError(f"no {name} robot in the archive")

    def insert(self, elite: Elite, parent: Optional[str] = None) -> bool:
        grid = self._ensure(len(elite.scores))
        # an archive written before body_ground stores a narrower record
        width = grid.data()["behaviour"].shape[1]
        cell = cell_of(elite.behaviour, self.n_descriptors)
        eps = self.search.get("tie_parts", 0.0)
        incumbent = self.cells.get(cell) if eps > 0 else None   # off: no extra work
        if (incumbent is not None and np.isfinite(elite.fitness)
                and np.isfinite(incumbent.fitness)
                and elite.fitness <= incumbent.fitness
                and elite.fitness >= incumbent.fitness - eps * abs(incumbent.fitness)
                and len(morpho.decode(elite.morph).nodes)
                    < len(morpho.decode(incumbent.morph).nodes)):
            # ribs only replaces a cell on a strictly better objective; lower the
            # incumbent's threshold so this smaller body can take the cell instead
            grid._store._fields["threshold"][cell] = -np.inf
        info = grid.add(solution=[elite.morph], objective=[elite.fitness],
                        measures=[normalise(elite.behaviour, self.n_descriptors)],
                        ctrl=[elite.ctrl], scores=[elite.scores], born=[elite.born],
                        behaviour=[elite.behaviour[:width]])
        added = int(info["status"][0]) != 0
        if added:
            number = elite.uid
            if number < 0 or any(u == number for c, u in self.uids.items() if c != cell):
                number = self.next_uid          # new robot, or a uid another cell holds
            self.uids[cell] = number
            self.next_uid = max(self.next_uid, number + 1)
            uid = elite_id(elite.morph, elite.ctrl)
            if parent == uid:       # no self-loops in the family tree
                parent = None
            self.lineage[uid] = (
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
        old = state.get("body_model")
        if "cells" in state or (grid is not None and not isinstance(grid, CVTArchive)):
            raise ValueError("archive from an older body model, retrain")
        if old != BODY_MODEL and not _stale_ok:
            if migration_path(old):
                raise ValueError(
                    f"archive from body model {old}, this code is {BODY_MODEL}: "
                    "run experiments/migrate_archive.py to bring it up to date")
            raise ValueError("archive from an older body model, retrain")
        if grid is not None and not _stale_ok:
            axes = state.get("axes", "default")
            n_descriptors = AXES_SPEC.get(axes, AXES_SPEC["default"])[0]
            if grid.measure_dim != n_descriptors:
                raise ValueError(
                    f"archive with {grid.measure_dim} archive axes, this archive "
                    f"uses {n_descriptors} (--axes {axes}): retrain")
        self.__dict__.update(state)
        self.__dict__.setdefault("_grid", None)
        self.__dict__.setdefault("seconds", 8.0)   # old runs used 8 s episodes
        self.__dict__.setdefault("variants", 0)    # and no randomisation
        self.__dict__.setdefault("lineage", {})    # nor lineage records
        self.__dict__.setdefault("uids", {})       # nor stable robot ids
        self.__dict__.setdefault("next_uid", 0)
        self.__dict__.setdefault("task", "forward")   # nor tasks
        self.__dict__.setdefault("payload", 0.0)      # nor payload
        self.__dict__.setdefault("ablate", ())        # nor ablations
        self.__dict__.setdefault("rules", ())         # nor optional score rules
        self.__dict__.setdefault("score", {})         # nor score settings
        self.__dict__.setdefault("body", "")          # nor body options
        self.__dict__.setdefault("axes", "default")   # nor a different axes option
        self.__dict__.setdefault("locked", ())        # nor locked parts
        self.__dict__.setdefault("plugin", {})        # nor plugins
        self.__dict__.setdefault("search", {})        # nor search options
        if "profile" not in state:                    # nor hardware profiles
            self.__dict__["profile"] = hardware.default_profile()

    def best(self) -> Elite:
        return max(self.cells.values(), key=lambda e: e.fitness)

    def stats(self) -> dict:
        fits = [e.fitness for e in self.cells.values()]
        nov = self.novelty()
        return {"batch": self.batch, "evals": self.evals, "cells": len(fits),
                "best": max(fits), "mean": float(np.mean(fits)),
                "novelty": float(nov.mean()) if len(nov) else 0.0,
                # QD-score: sum of fitness shifted to be positive per cell
                "qd_score": float(sum(f + 5.0 for f in fits)),
                "symmetric": count_symmetric(self)}


def count_symmetric(archive: Archive) -> int:
    """Elites whose body is fully mirrored (morphology.symmetry is 1.0)."""
    return sum(1 for e in archive.cells.values()
               if morpho.symmetry(morpho.decode(e.morph)) == 1.0)


MERGE_KEYS = ("terrains", "seconds", "variants", "task", "payload", "ablate",
              "rules", "score", "body", "axes", "profile", "locked", "plugin")


def merge(archives: List[Archive]) -> Archive:
    """
    One archive from several runs of the same setup (islands): every cell keeps
    the best elite found by any of them. The CVT centroids are fixed
    (CVT_SEED), so a cell means the same in every run. Runs that resumed from a
    common archive share the start of their history; the merged history keeps
    that shared part plus one entry for the merge, and evals count the shared
    work once.
    """
    if not archives:
        raise ValueError("nothing to merge")
    first = archives[0]
    for a in archives:
        a.assign_uids()
    for a in archives[1:]:
        for k in MERGE_KEYS:
            x, y = getattr(a, k), getattr(first, k)
            if k == "profile":      # an older archive lacks the optional keys
                x, y = hardware.normalize(x), hardware.normalize(y)
            if x != y:
                raise ValueError(f"archives differ in {k}: "
                                 f"{getattr(first, k)!r} vs {getattr(a, k)!r}")
    shared = 0
    while (all(len(a.history) > shared for a in archives)
           and all(a.history[shared] == first.history[shared] for a in archives)):
        shared += 1
    base_evals = first.history[shared - 1]["evals"] if shared else 0

    out = pickle.loads(pickle.dumps(first))
    out.assign_uids()
    # the first archive keeps its uids; a uid another island already holds in a
    # different cell gets a fresh one when its elite is inserted (Archive.insert)
    out.next_uid = max(out.next_uid, *(a.next_uid for a in archives))
    lineage: Dict[str, tuple] = {}
    for a in archives:
        lineage.update(a.lineage)
    for a in archives[1:]:
        for e in a.cells.values():
            out.insert(e, lineage.get(elite_id(e.morph, e.ctrl), (None,))[0])
    out.lineage = {**lineage, **out.lineage}
    out.batch = max(a.batch for a in archives)
    out.evals = base_evals + sum(a.evals - base_evals for a in archives)
    s = out.stats()
    s.update(added=0, seconds=0.0, merged=len(archives))
    out.history = first.history[:shared] + [s]
    return out


def code_version() -> str:
    """Commit of the code running this, for the archive history: `git archive`
    writes it into darwinbot/COMMIT (export-subst in .gitattributes), so a copy
    without .git (a Kaggle dataset) still knows it; in a checkout git answers,
    with "-dirty" when files differ from the commit; "unknown" otherwise."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "COMMIT")) as f:
            text = f.read().strip()
        if text and not text.startswith("$Format"):
            return text
    except OSError:
        pass
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, text=True,
                              capture_output=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=here, text=True, capture_output=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _init_worker(ablations=(), score_rules=(), body="", plugin_paths=(), score=None):
    """Pool workers leave Ctrl+C to the main process, which stops them, and
    carry the run's ablations, optional score rules, body options and plugins.
    A score dict is applied when given; callers with no archive keep their own."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ablate.set_active(ablations)
    rules.set_active(score_rules)
    body_options.set_active(body)
    if score is not None:
        apply_score(score)
    plugins.load_all(plugin_paths)


def _cuda_devices() -> int:
    import warp as wp
    return max(1, wp.get_cuda_device_count())


def _warp_worker(counter, n_gpus: int, ablations=(), score_rules=(), body="",
                 plugin_paths=(), score=None):
    """Pool initializer for --device warp: bind this worker to the next GPU."""
    _init_worker(ablations, score_rules, body, plugin_paths, score)
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    import warp as wp
    if wp.get_cuda_device_count() > 0:
        wp.set_device(f"cuda:{i % n_gpus}")


GRAFT_P = 0.2           # share of children made by grafting a limb of a second elite
LOCK_TRIES = 8          # offspring drawn until one keeps the archive's locked parts
BUDGET_JOINTS = 4       # bodies with more joints get proportionally more evaluations
BUDGET_SCALE_MAX = 3.0
PROTECT_SLOTS = 2       # changed slots that make a child's body a new body


def scaled_budget(budget: int, morph) -> int:
    """CMA-ES evaluations for a body: every joint adds CTRL_PER_NODE controller
    genes, so a body with many joints gets up to BUDGET_SCALE_MAX times the base budget
    to learn a coordinated gait instead of losing its cell to a simpler one."""
    scale = min(max(morph.n_joints / BUDGET_JOINTS, 1.0), BUDGET_SCALE_MAX)
    return int(round(budget * scale))


def child_budget(m: dict, budget: int, protect_new: float = 1.0) -> int:
    """CMA-ES evaluations of one child, before `scaled_budget`: a body that
    changed a lot (a graft, PROTECT_SLOTS or more slots of the genome, or
    another set of node slots) gets its budget times protect_new, so a new body
    is not judged on a controller tuned for its parent (Cheney et al. 2018).
    The child that is the parent's own body ("same") and random bodies keep
    their budget; 1.0 turns the option off and returns the budget untouched."""
    if protect_new == 1.0 or m["changed_slots"] is None or m["origin"] == "same":
        return budget
    big = (m["origin"] == "graft" or m["changed_slots"] >= PROTECT_SLOTS
           or not m["same_slots"])
    return int(round(budget * protect_new)) if big else budget


SEARCH_FLAGS = {"parent_selection": "--parent-selection", "inner_strategy": "--inner-strategy",
                "variant_mode": "--variant-mode", "protect_new": "--protect-new",
                "mirror_p": "--mirror-p", "mutation_bias": "--mutation-bias",
                "tie_parts": "--tie-parts", "batch_size": "--batch-size",
                "budget_init": "--budget-init", "budget_child": "--budget-child",
                "deterministic": "--deterministic",
                "screen": "--screen"}           # Archive.search keys and their train.py flags


def _shown(value):
    return "default" if value is None else value


def offspring(parent, donor, grafted: bool, rng, mirror_p: float = 0.0,
              mutation_bias: Optional[str] = None) -> tuple:
    """(child, origin) of one offspring: a graft of a donor's limb, or a mutation
    of the parent. A graft that fits no slot gives the parent's body back, so it
    is a plain mutation."""
    if grafted:
        child = morpho.graft(parent, donor, rng)
        if not np.array_equal(child, parent):
            return child, "graft"
    kwargs = {"mirror_p": mirror_p}
    if mutation_bias is not None:       # callers without a bias keep the old call
        kwargs["mutation_bias"] = mutation_bias
    return morpho.mutate(parent, rng, **kwargs), "mutate"


SCREENED_ORIGINS = ("mutate", "graft")


def job_cost(job) -> float:
    """Expected cost of a pool job, relative to the others of its batch: the
    CMA-ES evaluations (`scaled_budget`) times the actuators, which set the
    time per evaluation. Only used to order the jobs of a batch."""
    morph = morpho.decode(job[0])
    return scaled_budget(job[3], morph) * max(len(morph.layout()), 1)


def longest_first(tasks: list) -> list:
    """The jobs of a batch with the most expensive first, so the last ones to
    finish are short and the workers wait less for the slowest body. Stable for
    equal costs. Only the submission order changes: every result carries its
    body, and the archive already takes results in completion order."""
    return sorted(tasks, key=job_cost, reverse=True)


def _work(job):
    """Pool worker: optimise one body's controller. Runs in a subprocess.
    An exception in one body (a compile error, a numerical failure) must not end
    a long run: it is printed with a short id of the body and the result comes
    back with evals == 0 and the failed flag set, which the loop counts."""
    (morph_genes, x0, terrains, budget, sigma0, seed, seconds,
     device, popsize, variants, parent, task, payload) = job[:13]
    opts = job[13] if len(job) > 13 else {}     # inner-loop options of this job
    morpho.set_mutation_bias(opts.get("mutation_bias", "default"))
    tdir = trace.directory()      # tracing: a 15th job element is the batch index
    t0 = time.time()
    rec = trace.Recorder() if tdir else None
    extra = None
    try:
        morph = morpho.decode(morph_genes)
        terr = get_terrains(terrains, seed=seed)
        res = optimise_controller(morph, terr, x0=x0,
                                  budget=scaled_budget(budget, morph), sigma0=sigma0,
                                  seed=seed, seconds=seconds, device=device,
                                  popsize=popsize, variants=variants, task=task,
                                  payload=payload, recorder=rec,
                                  strategy=opts.get("strategy"),
                                  parent_genes=opts.get("parent_genes"),
                                  variant_mode=opts.get("variant_mode", "all"))
        if tdir:
            extra = {"idx": job[14] if len(job) > 14 else None,
                     "id": trace.body_id(morph_genes), "file": None,
                     "budget": scaled_budget(budget, morph),
                     "seconds": round(time.time() - t0, 3)}
            try:
                if res.evals > 0 and rec.runs:
                    extra["file"] = rec.write(
                        tdir, morph_genes, x0, ctrl.active_genes(morph),
                        extra["budget"], sigma0, seed, seconds, variants, task,
                        [t.name for t in terr], commands(task), task_score)
            except Exception:
                print(f"trace write failed:\n{traceback.format_exc()}", flush=True)
    except Exception:
        body = hashlib.sha1(np.asarray(morph_genes, float).tobytes()).hexdigest()[:8]
        print(f"body {body} failed:\n{traceback.format_exc()}", flush=True)
        out = (morph_genes, x0, -np.inf, [], 0, None, parent, True)
        return out + (None,) if tdir else out
    out = (morph_genes, res.ctrl, res.fitness, res.scores, res.evals,
           res.behaviour, parent, False)
    return out + (extra,) if tdir else out


def _screen_work(job):
    """Pool worker: quick score of one changed body, the controller it inherits
    on one nominal flat episode of the "forward" task. -inf if unbuildable or if
    the episode fails."""
    i, genes, ctrl_, seconds, payload = job
    try:
        ep = Episode(morpho.decode(genes), flat(), seconds, payload)
        if not ep.buildable:
            return i, -np.inf
        return i, float(evaluate([ep], ctrl_, None, "forward")[0])
    except Exception:
        body = hashlib.sha1(np.asarray(genes, float).tobytes()).hexdigest()[:8]
        print(f"screen of body {body} failed:\n{traceback.format_exc()}", flush=True)
        return i, -np.inf


def screen_keep(origins: list, scores: dict, keep: float) -> set:
    """Indices that survive the screen: every body that was not screened (origin
    not mutate or graft) and the best ceil(keep * n) of the n screened ones by
    quick score (ties: batch order)."""
    screened = [i for i, o in enumerate(origins) if o in SCREENED_ORIGINS]
    best = sorted(screened, key=lambda i: (-scores[i], i))[:math.ceil(keep * len(screened))]
    return (set(range(len(origins))) - set(screened)) | set(best)


def _trace_body(tdir, archive, batch_meta, res, cell, before, accepted):
    """Tracing: one line of bodies.jsonl for a finished body (main process)."""
    genes, ctrl_, fit, scores, evals, behaviour, parent, bad = res[:8]
    extra = res[8] if len(res) > 8 and res[8] else {}
    idx = extra.get("idx")
    m = batch_meta[idx] if idx is not None and idx < len(batch_meta) else {}
    rec = {"batch": archive.batch, "id": trace.body_id(genes),
           "file": extra.get("file"), "parent": parent,
           "origin": m.get("origin"), "changed_slots": m.get("changed_slots"),
           "l2": m.get("l2"), "budget": extra.get("budget"), "evals": int(evals),
           "seconds": extra.get("seconds"),
           "fitness": None if not np.isfinite(fit) else float(fit),
           "behaviour": None if behaviour is None else [float(v) for v in behaviour],
           "cell": cell, "before": before, "accepted": accepted, "failed": bool(bad)}
    with open(os.path.join(tdir, "bodies.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(
    terrains: str = "flat",
    n_init: int = 24,
    n_batches: int = 30,
    batch_size: int = 16,
    budget_init: int = 400,
    budget_child: int = 120,
    episode_seconds: float = EPISODE_SECONDS,
    random_frac: float = 0.1,
    workers: Optional[int] = None,
    seed: int = 0,
    out_dir: str = "checkpoints",
    resume: Optional[str] = None,
    device: str = "cpu",
    popsize: Optional[int] = None,
    variants: int = 2,
    parent_selection: str = "novelty",
    task: str = "forward",
    payload: float = 0.0,
    ablations: tuple = (),
    score_rules: Optional[tuple] = None,   # None: not given, so a resume keeps the archive's
    score: Optional[dict] = None,          # new archives only; a resume keeps its own
    body: str = "",
    axes: str = "default",
    plugin: Optional[dict] = None,
    inner_strategy: Optional[str] = None,
    variant_mode: str = "all",
    protect_new: float = 1.0,
    mirror_p: float = 0.0,
    mutation_bias: str = "default",
    tie_parts: float = 0.0,
    deterministic: bool = False,
    screen: float = 1.0,
) -> Archive:
    if parent_selection not in PARENT_SELECTIONS:
        raise ValueError(f"parent_selection must be one of {PARENT_SELECTIONS}")
    if inner_strategy not in INNER_STRATEGIES:
        raise ValueError(f"inner_strategy must be one of {INNER_STRATEGIES}")
    if variant_mode not in VARIANT_MODES:
        raise ValueError(f"variant_mode must be one of {VARIANT_MODES}")
    if variant_mode == "first" and device == "warp":
        raise ValueError("--variant-mode first is CPU only; use all with --device warp")
    if protect_new < 1.0:
        raise ValueError("protect_new must be >= 1.0 (1.0 turns it off)")
    if not 0.0 <= mirror_p <= 1.0:
        raise ValueError("mirror_p must be in [0, 1] (0 turns it off)")
    morpho.set_mutation_bias(mutation_bias)     # also validates the name
    if tie_parts < 0.0:
        raise ValueError("tie_parts must be >= 0 (0 turns it off)")
    if not 0.0 < screen <= 1.0:
        raise ValueError("screen must be in (0, 1] (1.0 turns it off)")
    if screen < 1.0 and device in ("gpu", "warp"):
        raise ValueError(f"--screen is CPU only; use --screen 1 with --device {device}")
    if device in ("gpu", "warp"):
        # sim_gpu and sim_warp do not simulate the oscillator coupling yet
        # (BODY_MODEL 24); every controller drawn here carries it.
        raise ValueError(f"--device {device} does not simulate coupled oscillators yet; use --device cpu")
    if axes not in AXES:
        raise ValueError(f"axes must be one of {AXES}")
    if task not in task_names():
        raise ValueError(f"task must be one of {task_names()}")
    if device == "gpu" and variants > 0 and not resume:
        raise ValueError("randomisation is CPU only; use --variants 0 with --device gpu")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    if device == "gpu":
        print("[darwinbot] gpu: workers ignored, tasks run one at a time")
        workers = 1
        popsize = 64 if popsize is None else popsize
    elif device == "warp":
        # One process per worker, spread over the GPUs; by default one per GPU.
        workers = workers or _cuda_devices()
    else:
        workers = workers or max(1, multiprocessing.cpu_count() - 1)

    if resume:
        archive: Archive = load_archive(resume)
        if axes != "default" and axes != archive.axes:
            raise ValueError(f"{resume} uses --axes {archive.axes}; --axes {axes} "
                             "applies to a new archive only")
        if score_rules is not None and rules.normalise(score_rules) != archive.rules:
            raise ValueError(f"{resume} uses --rules {','.join(archive.rules) or 'none'}, "
                             f"not {','.join(rules.normalise(score_rules)) or 'none'}; "
                             "--rules applies to a new archive only")
        terrains = archive.terrains     # task and payload also come from the archive
        rng = np.random.default_rng(seed + archive.batch)
    else:
        archive = Archive(terrains=terrains, seconds=episode_seconds,
                          variants=variants, task=task, payload=payload,
                          ablate=ablate.normalise(ablations),
                          rules=rules.normalise(score_rules),
                          score=dict(score or {}),
                          body=body_options.parse(body).to_string(),
                          axes=axes, plugin=plugin or {})
    search = {"parent_selection": parent_selection, "inner_strategy": inner_strategy,
              "variant_mode": variant_mode, "protect_new": protect_new,
              "mirror_p": mirror_p, "mutation_bias": mutation_bias,
              "tie_parts": tie_parts, "batch_size": batch_size,
              "budget_init": budget_init, "budget_child": budget_child,
              "deterministic": deterministic,
              "screen": screen}
    if resume:                                  # the new values win, and are stored
        for key, value in search.items():
            old = archive.search.get(key, value)
            if old != value:
                print(f"resume: {SEARCH_FLAGS[key]} {_shown(value)} differs from the "
                      f"archive's {_shown(old)}")
    archive.search = search
    ablate.set_active(archive.ablate)           # a resume keeps the run's ablations
    rules.set_active(archive.rules)             # and its score rules
    body_options.set_active(archive.body)       # and its body limits
    apply_score(archive.score)                  # and its score settings
    if device == "gpu" and archive.variants > 0:
        raise ValueError("randomisation is CPU only; use --variants 0 with --device gpu")

    def new_seed():
        return int(rng.integers(0, 2**31 - 1))

    tdir = trace.directory()
    batch_meta: list = []      # tracing: origin and distance to the parent of each job

    def meta(child, parent=None, origin="random"):
        if parent is None:
            return {"origin": origin, "changed_slots": None, "l2": None}
        d = (np.asarray(child, float) - np.asarray(parent, float)).reshape(
            morpho.MAX_NODES, morpho.BLOCK_GENES)
        return {"origin": origin, "changed_slots": int((np.abs(d).max(axis=1) > 0).sum()),
                "l2": float(np.sqrt((d ** 2).sum())),
                "same_slots": set(morpho.decode(child).nodes)
                == set(morpho.decode(parent).nodes)}

    def inner_opts(parent_genes=None) -> dict:
        """Inner-loop options of one job; focused needs the parent's body genes."""
        return {"strategy": inner_strategy, "variant_mode": variant_mode,
                "mutation_bias": mutation_bias,
                "parent_genes": None if parent_genes is None else np.array(parent_genes)}

    def make_batch() -> list:
        batch_meta.clear()
        tasks = _make_batch()
        if tdir:
            tasks = [t + (i,) for i, t in enumerate(tasks)]
        return tasks

    def _make_batch() -> list:
        if not archive.cells:
            tasks = [(morpho.random_morphology(rng), None, terrains,
                      budget_init, 0.25, new_seed(), archive.seconds,
                      device, popsize, archive.variants, None,
                      archive.task, archive.payload, inner_opts(None))
                     for _ in range(n_init)]
            batch_meta.extend(meta(t[0]) for t in tasks)
            return tasks
        parents = archive.sample_elites(batch_size, rng, parent_selection)
        tasks = []
        for i in range(batch_size):
            # A random body would not carry the locked parts.
            if not archive.locked and rng.random() < random_frac:
                tasks.append((morpho.random_morphology(rng), None, terrains,
                              budget_init, 0.25, new_seed(), archive.seconds,
                              device, popsize, archive.variants, None,
                              archive.task, archive.payload, inner_opts(None)))
                batch_meta.append(meta(tasks[-1][0]))
            else:
                parent = parents["solution"][i]
                donor = parents["solution"][(i + 1) % batch_size]
                for _ in range(LOCK_TRIES):
                    child, origin = offspring(parent, donor, rng.random() < GRAFT_P,
                                              rng, mirror_p, mutation_bias)
                    if morpho.keeps_locked(child, parent, archive.locked):
                        break
                else:                    # same body, its controller searched again
                    child, origin = np.array(parent), "same"
                batch_meta.append(meta(child, parent, origin))
                tasks.append((child, parents["ctrl"][i], terrains,
                              child_budget(batch_meta[-1], budget_child, protect_new),
                              0.12, new_seed(), archive.seconds,
                              device, popsize, archive.variants,
                              elite_id(parents["solution"][i], parents["ctrl"][i]),
                              archive.task, archive.payload, inner_opts(parent)))
        return tasks

    print(f"[darwinbot] terrains={terrains} workers={workers} device={device} "
          f"batches={n_batches} batch_size={batch_size} "
          f"task={archive.task} payload={archive.payload} kg"
          + (f" ablate={list(archive.ablate)}" if archive.ablate else "")
          + (f" rules={list(archive.rules)}" if archive.rules else "")
          + (f" score={archive.score}" if archive.score else "")
          + (f" body={archive.body}" if archive.body else "")
          + (f" axes={archive.axes}" if archive.axes != "default" else "")
          + (f" inner_strategy={inner_strategy}" if inner_strategy else "")
          + (f" variant_mode={variant_mode}" if variant_mode != "all" else "")
          + (f" protect_new={protect_new}" if protect_new != 1.0 else "")
          + (f" mirror_p={mirror_p}" if mirror_p else "")
          + (f" mutation_bias={mutation_bias}" if mutation_bias != "default" else "")
          + (f" tie_parts={tie_parts}" if tie_parts else "")
          + (f" screen={screen}" if screen != 1.0 else "")
          + (" deterministic" if deterministic else ""))
    if device == "gpu" or (device == "warp" and workers == 1):
        pool = None
    elif device == "warp":
        # CUDA does not survive fork: spawned workers, each bound to one GPU.
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(workers, _warp_worker,
                        (ctx.Value("i", 0), _cuda_devices(), archive.ablate, archive.rules,
                         archive.body, plugins.loaded(), archive.score))
    else:
        fastloop.warmup()        # compiled before the fork, so workers inherit it
        pool = multiprocessing.Pool(workers, _init_worker,
                                    (archive.ablate, archive.rules, archive.body,
                                     plugins.loaded(), archive.score))
    if tdir:
        os.makedirs(tdir, exist_ok=True)
    path = os.path.join(out_dir, "archive.pkl")
    code = code_version()
    print(f"[darwinbot] code {code}")
    saved = None                                  # batch count in the file on disk
    try:
        left, seeding_tries = n_batches, 0
        # Until the archive has a body, batches are random and do not count;
        # that happens at start, or when every body of the first batch was
        # unbuildable.
        while left > 0 or (not archive.cells and seeding_tries < MAX_SEED_TRIES):
            if archive.cells:
                left -= 1
            else:
                seeding_tries += 1
            t0 = time.time()
            added = failed = 0
            tasks = make_batch()
            n_screened = 0
            if screen < 1.0 and any(m["origin"] in SCREENED_ORIGINS for m in batch_meta):
                # quick score of the changed bodies on the same pool; the
                # results are matched by index, so the order does not matter
                jobs = [(i, t[0], t[1], t[6], t[12]) for i, t in enumerate(tasks)
                        if batch_meta[i]["origin"] in SCREENED_ORIGINS]
                quick = (map(_screen_work, jobs) if pool is None else
                         (pool.imap if deterministic else pool.imap_unordered)(
                             _screen_work, jobs))
                kept = screen_keep([m["origin"] for m in batch_meta],
                                   dict(quick), screen)
                n_screened = len(tasks) - len(kept)
                tasks = [t for i, t in enumerate(tasks) if i in kept]
            # deterministic: results in submission order, so the same seed and
            # worker count give the same run; otherwise in completion order
            results = ([_work(t) for t in tasks] if pool is None
                       else (pool.imap if deterministic else pool.imap_unordered)(
                           _work, longest_first(tasks)))
            for res in results:
                genes, ctrl, fit, scores, evals, behaviour, parent, bad = res[:8]
                archive.evals += evals
                failed += bad
                if evals == 0:      # unbuildable: it would only occupy a cell
                    if tdir:
                        _trace_body(tdir, archive, batch_meta, res, None, None, False)
                    continue
                cell = cell_of(behaviour, archive.n_descriptors) if tdir else None
                before = None
                if tdir:
                    inc = archive.cells.get(cell)
                    before = None if inc is None else float(inc.fitness)
                ok = archive.insert(Elite(genes, ctrl, fit, scores,
                                          archive.batch, behaviour), parent)
                added += ok
                if tdir:
                    _trace_body(tdir, archive, batch_meta, res, cell, before, bool(ok))
            if tasks and failed == len(tasks):     # a systematic bug, not one odd body
                raise RuntimeError(f"all {failed} bodies of the batch failed")
            if not archive.cells:
                print("batch with no buildable body, drawing a new random one",
                      flush=True)
                continue
            s = archive.stats()
            s.update(added=added, seconds=round(time.time() - t0, 1), code=code)
            archive.history.append(s)
            print(f"batch {s['batch']:3d}  cells {s['cells']:3d}  +{added:2d}  "
                  f"best {s['best']:6.2f}  mean {s['mean']:6.2f}  "
                  f"nov {s['novelty']:.3f}  evals {s['evals']}  {s['seconds']}s"
                  + (f"  screened={n_screened}" if screen < 1.0 else "")
                  + (f"  failed {failed}" if failed else ""), flush=True)
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
        archive.interrupted = True     # lets a caller running several presets stop too
        return archive
    if pool is not None:
        pool.close()
        pool.join()
    return archive

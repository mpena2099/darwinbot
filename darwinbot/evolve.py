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

New bodies are mutated copies of elites drawn with probability proportional to
their novelty (mean distance to the k nearest elites in descriptor space), or
uniformly with parent_selection="uniform". The child starts its inner
optimisation from the parent's controller (Lamarckian inheritance), which is
much cheaper than learning a gait from scratch.
"""

import copy
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

from darwinbot import ablate, body_options, hardware, plugins, rules
from darwinbot import controller as ctrl
from darwinbot import morphology as morpho
from darwinbot.fitness import task_names
from darwinbot.inner import optimise_controller
from darwinbot.sim import COT_MAX, COT_MIN, HEIGHT_MAX
from darwinbot.terrain import get_terrains

BODY_MODEL = 23                     # bump when the genome meaning changes
N_CELLS    = 800                    # CVT cells (five axes: 500 would be too coarse)
CVT_SEED   = 0                      # fixed seed: same centroids in every run
K_NOVELTY  = 10                     # neighbours in the novelty score
MAX_SEED_TRIES = 20                 # random batches tried to fill an empty archive
PARENT_SELECTIONS = ("novelty", "uniform")

# Name bins. Height is torso z over body reach, duty is the contact fraction.
HEIGHT_BINS = ((0.4, "low"), (0.8, "mid"), (float("inf"), "high"))
ROLL_SHARE  = 0.5                   # wheel share of contact time above this: roll
CRAWL_DUTY, WALK_DUTY, BELLY_DUTY = 0.8, 0.5, 0.15

Cell = int                          # index of the CVT cell


N_DESCRIPTORS = 5


def normalise(behaviour) -> np.ndarray:
    """Descriptor in [0, 1]^5 (duty, height, servos, legs, cot) from a behaviour vector, so that every axis
    weighs the same. Cost of transport spans decades, so it is taken in log."""
    duty, height, servos, _, cot, legs = np.asarray(behaviour, dtype=float)[:6]
    cot = np.log(np.clip(cot, COT_MIN, COT_MAX) / COT_MIN) / np.log(COT_MAX / COT_MIN)
    return np.clip([duty, height / HEIGHT_MAX, servos / morpho.MAX_SERVOS,
                    legs / morpho.MAX_LEGS, cot], 0.0, 1.0)


@lru_cache(maxsize=1)
def _centroids() -> np.ndarray:
    probe = CVTArchive(solution_dim=1, centroids=N_CELLS, ranges=[(0.0, 1.0)] * N_DESCRIPTORS,
                       seed=CVT_SEED, samples=20000)
    return np.array(probe.centroids)


def cell_of(behaviour) -> Cell:
    """CVT cell of a behaviour vector (duty, height, servos, wheel share, cot, legs)."""
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
    # (duty, height, servos, wheel share, cot, legs); None: a body never simulated
    behaviour: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.behaviour is None:
            m = morpho.decode(self.morph)
            self.behaviour = np.array(
                [0.0, 0.0, float(m.n_servos), 0.0, COT_MAX, float(m.n_legs)])
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
    task:     str = "forward"  # fitness.TASKS: how episodes are scored
    payload:  float = 0.0      # extra torso mass in kg
    ablate:   tuple = ()       # ablate.NAMES switched off in this run (experiments)
    rules:    tuple = ()       # rules.NAMES switched on in this run (experiments)
    body:     str = ""         # body_options limits, as given to train.py --body
    profile:  dict = field(default_factory=lambda: copy.deepcopy(hardware.PROFILE))
    locked:   tuple = ()       # node ids mutation must keep (0: the torso), from seed.py
    plugin:   dict = field(default_factory=dict)   # name and sha256 of the --plugin file
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
                    "behaviour": ((6,), np.float64),
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

    def like(self) -> "Archive":
        """Empty archive with this run's settings, but no cells, history or lineage."""
        return Archive(terrains=self.terrains, batch=self.batch, evals=self.evals,
                       seconds=self.seconds, variants=self.variants, task=self.task,
                       payload=self.payload, ablate=self.ablate, rules=self.rules,
                       body=self.body, profile=copy.deepcopy(self.profile),
                       locked=self.locked, plugin=dict(self.plugin))

    def activate(self) -> None:
        """Turn on the ablations, score rules and body limits this archive was
        trained with; scripts that score, decode or build its robots call this."""
        ablate.set_active(self.ablate)
        rules.set_active(self.rules)
        body_options.set_active(self.body)

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
            stored = names.get(index, "")
            # a unique name carries no cell suffix, so both spellings name the cell
            if stored != base and stored.rsplit("-", 1)[0] != base:
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
        if (state.get("body_model") != BODY_MODEL or "cells" in state
                or (grid is not None and not isinstance(grid, CVTArchive))):
            raise ValueError("archive from an older body model, retrain")
        if grid is not None and grid.measure_dim != N_DESCRIPTORS:
            raise ValueError(
                f"archive with {grid.measure_dim} archive axes, this code uses "
                f"{N_DESCRIPTORS} (the servo count is an axis again): retrain")
        self.__dict__.update(state)
        self.__dict__.setdefault("_grid", None)
        self.__dict__.setdefault("seconds", 8.0)   # old runs used 8 s episodes
        self.__dict__.setdefault("variants", 0)    # and no randomisation
        self.__dict__.setdefault("lineage", {})    # nor lineage records
        self.__dict__.setdefault("task", "forward")   # nor tasks
        self.__dict__.setdefault("payload", 0.0)      # nor payload
        self.__dict__.setdefault("ablate", ())        # nor ablations
        self.__dict__.setdefault("rules", ())         # nor optional score rules
        self.__dict__.setdefault("body", "")          # nor body options
        self.__dict__.setdefault("locked", ())        # nor locked parts
        self.__dict__.setdefault("plugin", {})        # nor plugins
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
                "qd_score": float(sum(f + 5.0 for f in fits))}


MERGE_KEYS = ("terrains", "seconds", "variants", "task", "payload", "ablate",
              "rules", "body", "profile", "locked", "plugin")


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
    for a in archives[1:]:
        for k in MERGE_KEYS:
            if getattr(a, k) != getattr(first, k):
                raise ValueError(f"archives differ in {k}: "
                                 f"{getattr(first, k)!r} vs {getattr(a, k)!r}")
    shared = 0
    while (all(len(a.history) > shared for a in archives)
           and all(a.history[shared] == first.history[shared] for a in archives)):
        shared += 1
    base_evals = first.history[shared - 1]["evals"] if shared else 0

    out = pickle.loads(pickle.dumps(first))
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


def _init_worker(ablations=(), score_rules=(), body="", plugin_paths=()):
    """Pool workers leave Ctrl+C to the main process, which stops them, and
    carry the run's ablations, optional score rules, body options and plugins."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ablate.set_active(ablations)
    rules.set_active(score_rules)
    body_options.set_active(body)
    plugins.load_all(plugin_paths)


GRAFT_P = 0.2           # share of children made by grafting a limb of a second elite
LOCK_TRIES = 8          # offspring drawn until one keeps the archive's locked parts
BUDGET_JOINTS = 4       # bodies with more joints get proportionally more evaluations
BUDGET_SCALE_MAX = 3.0


def scaled_budget(budget: int, morph) -> int:
    """CMA-ES evaluations for a body: every joint adds six controller genes, so
    a body with many joints gets up to BUDGET_SCALE_MAX times the base budget
    to learn a coordinated gait instead of losing its cell to a simpler one."""
    scale = min(max(morph.n_joints / BUDGET_JOINTS, 1.0), BUDGET_SCALE_MAX)
    return int(round(budget * scale))


def _work(job):
    """Pool worker: optimise one body's controller. Runs in a subprocess."""
    (morph_genes, x0, terrains, budget, sigma0, seed, seconds,
     device, popsize, variants, parent, task, payload) = job
    morph = morpho.decode(morph_genes)
    res = optimise_controller(morph, get_terrains(terrains, seed=seed), x0=x0,
                              budget=scaled_budget(budget, morph), sigma0=sigma0, seed=seed,
                              seconds=seconds, device=device, popsize=popsize,
                              variants=variants, task=task, payload=payload)
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
    task: str = "forward",
    payload: float = 0.0,
    ablations: tuple = (),
    score_rules: tuple = (),
    body: str = "",
    plugin: Optional[dict] = None,
) -> Archive:
    if parent_selection not in PARENT_SELECTIONS:
        raise ValueError(f"parent_selection must be one of {PARENT_SELECTIONS}")
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
    else:
        workers = workers or max(1, multiprocessing.cpu_count() - 1)

    if resume:
        with open(resume, "rb") as f:
            archive: Archive = pickle.load(f)
        terrains = archive.terrains     # task and payload also come from the archive
        rng = np.random.default_rng(seed + archive.batch)
    else:
        archive = Archive(terrains=terrains, seconds=episode_seconds,
                          variants=variants, task=task, payload=payload,
                          ablate=ablate.normalise(ablations),
                          rules=rules.normalise(score_rules),
                          body=body_options.parse(body).to_string(),
                          plugin=plugin or {})
    ablate.set_active(archive.ablate)           # a resume keeps the run's ablations
    rules.set_active(archive.rules)             # and its score rules
    body_options.set_active(archive.body)       # and its body limits
    if device == "gpu" and archive.variants > 0:
        raise ValueError("randomisation is CPU only; use --variants 0 with --device gpu")

    def new_seed():
        return int(rng.integers(0, 2**31 - 1))

    def make_batch() -> list:
        if not archive.cells:
            return [(morpho.random_morphology(rng), None, terrains,
                     budget_init, 0.25, new_seed(), archive.seconds,
                     device, popsize, archive.variants, None,
                     archive.task, archive.payload) for _ in range(n_init)]
        parents = archive.sample_elites(batch_size, rng, parent_selection)
        tasks = []
        for i in range(batch_size):
            # A random body would not carry the locked parts.
            if not archive.locked and rng.random() < random_frac:
                tasks.append((morpho.random_morphology(rng), None, terrains,
                              budget_init, 0.25, new_seed(), archive.seconds,
                              device, popsize, archive.variants, None,
                              archive.task, archive.payload))
            else:
                parent = parents["solution"][i]
                for _ in range(LOCK_TRIES):
                    if rng.random() < GRAFT_P:        # crossover with another elite
                        donor = parents["solution"][(i + 1) % batch_size]
                        child = morpho.graft(parent, donor, rng)
                    else:
                        child = morpho.mutate(parent, rng)
                    if morpho.keeps_locked(child, parent, archive.locked):
                        break
                else:                    # same body, its controller searched again
                    child = np.array(parent)
                tasks.append((child, parents["ctrl"][i], terrains,
                              budget_child, 0.12, new_seed(), archive.seconds,
                              device, popsize, archive.variants,
                              elite_id(parents["solution"][i], parents["ctrl"][i]),
                              archive.task, archive.payload))
        return tasks

    print(f"[darwinbot] terrains={terrains} workers={workers} device={device} "
          f"batches={n_batches} batch_size={batch_size} "
          f"task={archive.task} payload={archive.payload} kg"
          + (f" ablate={list(archive.ablate)}" if archive.ablate else "")
          + (f" rules={list(archive.rules)}" if archive.rules else "")
          + (f" body={archive.body}" if archive.body else ""))
    pool = None if device == "gpu" else multiprocessing.Pool(
        workers, _init_worker,
        (archive.ablate, archive.rules, archive.body, plugins.loaded()))
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
            added = 0
            tasks = make_batch()
            results = ([_work(t) for t in tasks] if pool is None
                       else pool.imap_unordered(_work, tasks))
            for genes, ctrl, fit, scores, evals, behaviour, parent in results:
                archive.evals += evals
                if evals == 0:      # unbuildable: it would only occupy a cell
                    continue
                added += archive.insert(Elite(genes, ctrl, fit, scores,
                                              archive.batch, behaviour), parent)
            if not archive.cells:
                print("batch with no buildable body, drawing a new random one",
                      flush=True)
                continue
            s = archive.stats()
            s.update(added=added, seconds=round(time.time() - t0, 1), code=code)
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
        archive.interrupted = True     # lets a caller running several presets stop too
        return archive
    if pool is not None:
        pool.close()
        pool.join()
    return archive

"""
trace.py
--------
Optional trace of the inner loop for offline analysis (tools/speed/).

DARWINBOT_TRACE=<dir> switches it on: every body's inner loop writes one
compressed .npz (by the worker process) and the main loop appends one JSON line
per body to <dir>/bodies.jsonl. Unset, nothing here runs and nothing is
written. Tracing never changes a result: it only records what was computed.

npz layout (one file per body, name `<pid>-<counter>-<body id>.npz`):

  genes, x0 (empty when none), active          body genes, start controller, active controller gene indices
  budget, sigma0, seed, seconds, variants      the inner-loop settings (int / float scalars)
  task, terrains, commands                     task name, terrain names, [n, 2] (steering, speed)
  gen        [evals]                           CMA-ES generation (-1: the start controller)
  cand       [evals, len(active)] float32      candidate controller, active genes only
  fitness    [evals]
  scores     [evals, terrains]                 per-terrain score (mean over variants)
  task_scores [evals, terrains, variants]      task score of each terrain x variant
  metrics    [evals, terrains, variants, commands, fields] float32
  fields     names of the last axis of metrics
  seconds_wall                                 inner-loop wall seconds
"""

import hashlib
import itertools
import os
import time
from typing import List, Optional

import numpy as np

ENV = "DARWINBOT_TRACE"

FIELDS = ("dx", "dy", "progress", "yaw", "energy", "fell", "unstable", "duration",
          "duty", "height", "wheel_share", "mass", "upright", "tilt", "slip",
          "servo_ground")

_counter = itertools.count()


def directory() -> Optional[str]:
    """The trace directory, or None when tracing is off."""
    d = os.environ.get(ENV, "").strip()
    return d or None


def body_id(genes) -> str:
    """Short id of a body, the same in the npz name and in bodies.jsonl."""
    return hashlib.sha1(np.asarray(genes, float).tobytes()).hexdigest()[:8]


def metric_row(m) -> List[float]:
    return [float(getattr(m, f)) for f in FIELDS]


class Recorder:
    """Collects the evaluations of one body's inner loop (CPU, no process pool)."""

    def __init__(self):
        self.runs: list = []      # evaluate() appends [terrain][variant][command] Metrics
        self.gen = -1
        self._rows: list = []     # (gen, active genes, fitness, scores)
        self.t0 = time.time()

    def record(self, cand, fit, scores):
        self._rows.append((self.gen, np.asarray(cand, np.float32), float(fit),
                           [float(s) for s in scores]))

    def write(self, directory_: str, morph_genes, x0, active, budget, sigma0,
              seed, seconds, variants, task, terrains, commands, score_fn) -> str:
        """Write the npz; returns the file name."""
        os.makedirs(directory_, exist_ok=True)
        n = len(self._rows)
        metrics = np.array([[[[metric_row(m) for m in cmds] for cmds in vs]
                             for vs in per_terrain] for per_terrain in self.runs],
                           dtype=np.float32)
        task_scores = np.array([[[score_fn(cmds, task) for cmds in vs]
                                 for vs in per_terrain] for per_terrain in self.runs],
                               dtype=np.float32)
        name = f"{os.getpid()}-{next(_counter)}-{body_id(morph_genes)}.npz"
        tmp = os.path.join(directory_, name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            genes=np.asarray(morph_genes, float),
            x0=np.asarray([] if x0 is None else x0, float),
            active=np.asarray(active, int),
            budget=budget, sigma0=sigma0, seed=seed, seconds=seconds,
            variants=variants, task=task,
            terrains=np.array(list(terrains)), commands=np.asarray(commands, float),
            gen=np.array([r[0] for r in self._rows], int),
            cand=np.stack([r[1] for r in self._rows]) if n else np.zeros((0, len(active)), np.float32),
            fitness=np.array([r[2] for r in self._rows], float),
            scores=np.array([r[3] for r in self._rows], float),
            task_scores=task_scores, metrics=metrics,
            fields=np.array(FIELDS), seconds_wall=time.time() - self.t0)
        os.replace(tmp, os.path.join(directory_, name))
        return name

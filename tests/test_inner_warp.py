"""
test_inner_warp.py
------------------
Inner loop on device "warp": MuJoCo Warp searches the controllers, and the
result is scored again on the CPU exactly as inner.evaluate scores it, so the
archive never holds a GPU-only number. Skipped when mujoco_warp (the optional
warp extra) is not installed.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from darwinbot import evolve
from darwinbot import inner
from darwinbot import morphology as mo
from darwinbot import randomize
from darwinbot import terrain as te
from darwinbot.sim import Episode

# Warp falls back to its CPU device when there is no CUDA driver.
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
wp.set_device(DEVICE)


def _rescore(morph, terrains, ctrl, variants, seconds, task):
    """What the CPU branch of optimise_controller scores, recomputed here."""
    episodes = [Episode(morph, t, seconds=seconds) for t in terrains]
    var_sets = None
    if variants > 0:
        vrng = np.random.default_rng([3, 1])
        var_sets = [[randomize.perturb(ep.model, vrng) for _ in range(variants)]
                    for ep in episodes]
    return inner.evaluate(episodes, ctrl, var_sets, task)


def test_result_is_cpu_scored():
    morph = mo.template(2)
    terrains = te.get_terrains("flat")
    res = inner.optimise_controller(morph, terrains, budget=8, seconds=1.0,
                                    device="warp", seed=3, task="steer")
    fit, scores, metrics = _rescore(morph, terrains, res.ctrl, 0, 1.0, "steer")
    assert res.fitness == fit
    assert res.scores == scores
    assert [m.dx for m in res.metrics] == [m.dx for m in metrics]


def test_variants_are_the_cpu_ones():
    morph = mo.template(2)
    terrains = te.get_terrains("flat,rough", seed=0)
    res = inner.optimise_controller(morph, terrains, budget=6, seconds=1.0,
                                    device="warp", seed=3, task="steer",
                                    variants=1)
    fit, scores, metrics = _rescore(morph, terrains, res.ctrl, 1, 1.0, "steer")
    assert res.fitness == fit
    assert res.scores == scores
    assert [m.dx for m in res.metrics] == [m.dx for m in metrics]


def test_run_writes_archive(tmp_path):
    archive = evolve.run(n_init=2, n_batches=1, batch_size=2, budget_init=6,
                         budget_child=4, device="warp", out_dir=str(tmp_path),
                         task="steer", episode_seconds=1.0)
    assert (tmp_path / "archive.pkl").exists()
    assert archive.cells
    assert all(len(e.behaviour) == 6 for e in archive.cells.values())

"""
test_sim_gpu.py
---------------
MJX batch episodes against the CPU reference, plus a tiny GPU CMA-ES run.
Skipped when the optional jax/mjx extra is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from darwinbot import controller as co
from darwinbot import morphology as mo
from darwinbot import sim_gpu
from darwinbot import terrain as te
from darwinbot.inner import optimise_controller
from darwinbot.sim import Episode
from darwinbot.sim_gpu import BatchEpisode

# Four-legged body, no wheels.
MORPH = mo.Morphology(
    body_length=0.4, body_width=0.2,
    pairs=[mo.LegPair(i, x, 0.0, 0.10, 0.10, True, False, 0.05)
           for i, x in enumerate((0.4, -0.4))])


@pytest.fixture(scope="module")
def batch():
    return BatchEpisode(MORPH, te.flat(), seconds=1.0)


def test_batch_matches_cpu(batch):
    rng = np.random.default_rng(0)
    ctrls = rng.uniform(0.0, 1.0, (4, co.CTRL_LENGTH))
    cpu = Episode(MORPH, te.flat(), seconds=1.0)
    for got, ref in zip(batch.run_batch(ctrls), (cpu.run(c) for c in ctrls)):
        assert got.fell == ref.fell
        assert abs(got.dx - ref.dx) < 0.05, (got.dx, ref.dx)


def test_gpu_optimise_returns_finite(batch, monkeypatch):
    # Reuse the compiled batch above: one compile for the whole file.
    monkeypatch.setattr(sim_gpu, "BatchEpisode", lambda *a, **k: batch)
    res = optimise_controller(MORPH, [te.flat()], budget=8, popsize=4,
                              device="gpu", seconds=1.0)
    assert np.isfinite(res.fitness)

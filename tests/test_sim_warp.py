"""
test_sim_warp.py
----------------
MuJoCo Warp batch episodes against the CPU reference: the nominal body, the
per-world physics and latency of randomize.Variant, other terrains, and a
longer statistical run. Skipped when mujoco_warp (the optional warp extra) is
not installed.
"""

import dataclasses

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from darwinbot import controller as co
from darwinbot import morphology as mo
from darwinbot import randomize
from darwinbot import terrain as te
from darwinbot.sim import Episode
from darwinbot.sim_warp import WarpBatchEpisode

# Warp falls back to its CPU device when there is no CUDA driver.
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _ref(morph, ctrls, commands, seconds):
    cpu = Episode(morph, te.get_terrains("flat")[0], seconds=seconds)
    return [cpu.run(c, None, float(m[0]), float(m[1]))
            for c, m in zip(ctrls, commands)]


def _check(got, ref):
    """The tolerances of test_short_parity, world by world."""
    for i, (g, r) in enumerate(zip(got, ref)):
        assert abs(g.dx - r.dx) <= 0.01 + 0.05 * abs(r.dx), (i, g.dx, r.dx)
        assert abs(g.progress - r.progress) <= 0.02 + 0.05 * abs(r.progress), (i, g.progress, r.progress)
        assert abs(g.dy - r.dy) <= 0.01 + 0.05 * abs(r.dy), (i, g.dy, r.dy)
        assert abs(g.yaw - r.yaw) <= 0.05, (i, g.yaw, r.yaw)
        assert g.fell == r.fell, (i, g.fell, r.fell)
        assert g.unstable == r.unstable, (i, g.unstable, r.unstable)
        assert abs(g.energy - r.energy) <= 0.02 + 0.05 * r.energy, (i, g.energy, r.energy)
        assert abs(g.duty - r.duty) <= 0.1, (i, g.duty, r.duty)
        assert abs(g.height - r.height) <= 0.03, (i, g.height, r.height)
        assert abs(g.tilt - r.tilt) <= 0.05, (i, g.tilt, r.tilt)
        assert abs(g.slip - r.slip) <= 0.02 + 0.1 * r.slip, (i, g.slip, r.slip)
        assert abs(g.servo_ground - r.servo_ground) <= 0.1, (i, g.servo_ground, r.servo_ground)
        assert abs(g.duration - r.duration) <= 1e-4, (i, g.duration, r.duration)


@pytest.mark.parametrize("wheels", [False, True])
def test_short_parity(wheels):
    morph = mo.template(2, wheels=wheels)
    rng = np.random.default_rng(0)
    ctrls = np.stack([co.random_controller(rng) for _ in range(4)])
    commands = np.array([[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.5]])
    batch = WarpBatchEpisode(morph, te.get_terrains("flat")[0], seconds=1.0,
                             device=DEVICE)
    _check(batch.run_batch(ctrls, commands),
           _ref(morph, ctrls, commands, 1.0))


def test_variants_parity():
    morph = mo.template(2)
    terrain = te.get_terrains("flat")[0]
    rng = np.random.default_rng(2)
    ctrls = np.stack([co.random_controller(rng) for _ in range(6)])
    cpu = Episode(morph, terrain, seconds=1.0)
    variants = [randomize.perturb(cpu.model, np.random.default_rng(10 + i))
                for i in range(6)]
    # One world without delay (the ring must pass it through) and one at the
    # deepest delay of randomize.DEFAULT.
    variants[0] = dataclasses.replace(variants[0], latency=0)
    variants[1] = dataclasses.replace(variants[1], latency=2)
    batch = WarpBatchEpisode(morph, terrain, seconds=1.0, device=DEVICE)
    _check(batch.run_batch(ctrls, 0.0, variants),
           [cpu.run(c, v, 0.0, 1.0) for c, v in zip(ctrls, variants)])


@pytest.mark.parametrize("name", ["rubble", "steps", "soft", "downhill",
                                  "slippery"])
def test_terrains(name):
    morph = mo.template(2)
    terrain = te.get_terrains(name)[0]
    rng = np.random.default_rng(2)
    ctrls = np.stack([co.random_controller(rng) for _ in range(3)])
    cpu = Episode(morph, terrain, seconds=1.0)
    batch = WarpBatchEpisode(morph, terrain, seconds=1.0, device=DEVICE)
    _check(batch.run_batch(ctrls), [cpu.run(c, None, 0.0, 1.0) for c in ctrls])


def test_statistical_parity():
    morph = mo.template(2)
    rng = np.random.default_rng(1)
    ctrls = np.stack([co.random_controller(rng) for _ in range(12)])
    batch = WarpBatchEpisode(morph, te.get_terrains("flat")[0], seconds=3.0,
                             device=DEVICE)
    got = batch.run_batch(ctrls, 0.0)
    ref = _ref(morph, ctrls, np.tile([0.0, 1.0], (12, 1)), 3.0)
    warp_dx = np.array([m.dx for m in got])
    cpu_dx = np.array([m.dx for m in ref])
    fell = sum(g.fell == r.fell for g, r in zip(got, ref))
    assert np.corrcoef(warp_dx, cpu_dx)[0, 1] >= 0.9, (warp_dx, cpu_dx)
    assert fell >= 10, (fell, [g.fell for g in got], [r.fell for r in ref])


def test_unbuildable_body():
    # 820 mm links do not fit the print bed: inner.py reports such a body as an
    # unstable zero episode, and so does the batch.
    morph = mo.template(2, thigh=0.8, shin=0.8)
    batch = WarpBatchEpisode(morph, te.get_terrains("flat")[0], seconds=1.0,
                             device=DEVICE)
    assert not batch.buildable
    rng = np.random.default_rng(0)
    got = batch.run_batch(np.stack([co.random_controller(rng) for _ in range(3)]))
    for m in got:
        assert m.unstable and not m.fell
        assert (m.dx, m.dy, m.energy, m.duration, m.yaw) == (0.0,) * 5
        assert m.mass == batch.mass


def test_progress_rule_for_falls():
    """Metrics.progress as in Episode.run: 2 * (x_end - x_mid) once the half-way
    step was reached (a fall after it keeps that), 0 for a fall before it, dx
    for an episode that ended without either. The flags are written into the
    batch buffers after a normal run, so the rule is checked on its own."""
    morph = mo.template(2)
    batch = WarpBatchEpisode(morph, te.get_terrains("flat")[0], seconds=1.0,
                             device=DEVICE)
    rng = np.random.default_rng(0)
    ctrls = np.stack([co.random_controller(rng) for _ in range(3)])
    n, b = batch.start(ctrls)
    wp.synchronize()
    dx = b["dx"].numpy()
    b["fell"].assign(np.array([1, 1, 0], dtype=np.int32))
    b["reached"].assign(np.array([0, 1, 0] + [0] * (len(dx) - 3), dtype=np.int32)[:len(dx)])
    b["x_mid"].assign(np.where(np.arange(len(dx)) == 1, dx - 0.05, 0.0).astype(np.float32))
    got = batch._metrics(b, n)
    assert got[0].fell and got[0].progress == 0.0
    assert got[1].fell and got[1].progress == pytest.approx(0.1, abs=1e-5)
    assert not got[2].fell and got[2].progress == pytest.approx(dx[2], abs=1e-6)

"""Topology tests: volume, passive cells, compliance, polygons, validation."""

import numpy as np
import pytest

from darwinbot.topology import compliance, simp, to_polygons


def cantilever(ny=30, nx=80):
    """All solid, left column clamped, downward unit load on the right middle."""
    solid = np.ones((ny, nx), dtype=bool)
    keep = np.zeros_like(solid)
    keep[0:2, -2:] = True
    fixed = np.zeros_like(solid)
    fixed[:, 0] = True
    loads = [np.zeros((ny, nx, 2))]
    loads[0][ny // 2, -1, 1] = -1.0
    return solid, keep, fixed, loads


def test_cantilever():
    solid, keep, fixed, loads = cantilever()
    volfrac = 0.4
    density = simp(solid, keep, fixed, loads, volfrac=volfrac)
    assert density.shape == solid.shape
    assert np.all((density >= 0.0) & (density <= 1.0))
    assert density[solid].sum() / solid.sum() == pytest.approx(volfrac, rel=0.02)
    assert np.allclose(density[keep], 1.0)
    assert np.allclose(density[~solid], 1e-3)
    uniform = np.full(solid.shape, volfrac)
    uniform[keep] = 1.0
    assert compliance(density, solid, fixed, loads) < compliance(
        uniform, solid, fixed, loads)


def test_hole_stays_void():
    solid, keep, fixed, loads = cantilever()
    solid[10:16, 30:40] = False
    density = simp(solid, keep, fixed, loads, volfrac=0.5, iters=10)
    assert np.allclose(density[~solid], 1e-3)
    assert density[solid].sum() / solid.sum() == pytest.approx(0.5, rel=0.02)


def test_to_polygons_rectangle():
    density = np.zeros((10, 12))
    density[2:6, 3:9] = 1.0
    cell = 2.5
    polygons = to_polygons(density, cell)
    assert len(polygons) == 1
    polygon = polygons[0]
    assert polygon.ndim == 2 and polygon.shape[1] == 2
    assert np.allclose(polygon[0], polygon[-1])         # closed
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    assert lo[0] == pytest.approx(3 * cell, abs=cell)
    assert hi[0] == pytest.approx(9 * cell, abs=cell)
    assert lo[1] == pytest.approx(2 * cell, abs=cell)
    assert hi[1] == pytest.approx(6 * cell, abs=cell)


def test_invalid_inputs():
    ny, nx = 8, 10
    solid = np.ones((ny, nx), dtype=bool)
    empty = np.zeros((ny, nx), dtype=bool)
    loads = [np.zeros((ny, nx, 2))]
    with pytest.raises(ValueError):
        simp(solid, empty[:-1], empty, loads)
    with pytest.raises(ValueError):
        simp(solid, empty, empty[:-1], loads)
    for volfrac in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            simp(solid, empty, empty, loads, volfrac=volfrac)

"""
test_axes_r1.py
---------------
The release-1 archive axes: four descriptors in a 500-cell CVT grid, with no
servo-count axis. The default and body options stay unchanged.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from darwinbot import evolve
from darwinbot.sim import Behaviour


def _behaviour(servos=4):
    return Behaviour(0.5, 0.6, float(servos), 0.0, 10.0, 3.0)


def _elite(fitness, behaviour):
    return evolve.Elite(np.full(evolve.morpho.MORPH_LENGTH, 0.5),
                        np.zeros(evolve.ctrl.CTRL_LENGTH), fitness, [fitness], 0,
                        np.array(behaviour, dtype=float))


def test_r1_archive_has_500_cells_and_four_dimensions():
    archive = evolve.Archive(terrains="flat", axes="r1")
    assert archive.insert(_elite(1.0, _behaviour()))
    assert archive.axes == "r1"
    assert archive.n_descriptors == 4
    assert archive._grid.measure_dim == 4
    assert len(archive._grid.centroids) == 500
    assert evolve._centroids(4).shape == (500, 4)
    assert evolve._centroids(5).shape == (800, 5)    # cached separately


def test_r1_normalise_ignores_the_servo_count():
    few, many = _behaviour(4), _behaviour(16)
    assert np.allclose(evolve.normalise(few, 4), evolve.normalise(many, 4))
    assert evolve.cell_of(few, 4) == evolve.cell_of(many, 4)
    assert evolve.normalise(few, "r1").shape == (4,)
    # the default axes still keep the count as a separate descriptor
    assert not np.allclose(evolve.normalise(few, 5), evolve.normalise(many, 5))
    assert evolve.cell_of(few, 5) != evolve.cell_of(many, 5)


def test_r1_cell_mapping_uses_the_archive_axes():
    archive = evolve.Archive(terrains="flat", axes="r1")
    assert archive.insert(_elite(1.0, _behaviour(4)))
    cell = evolve.cell_of(_behaviour(4), 4)
    # same behaviour apart from the servos: the r1 cell is shared, the newer
    # robot replaces the older one, and its type name still shows the servos
    assert archive.insert(_elite(2.0, _behaviour(16)))
    assert len(archive.cells) == 1
    assert archive.cells[cell].fitness == 2.0
    assert archive.names()[cell] == "16servos-mid-walk-1"


def test_default_and_body_archives_are_unchanged():
    default = evolve.Archive(terrains="flat")
    assert default.insert(_elite(1.0, _behaviour()))
    assert default.n_descriptors == 5
    assert default._grid.measure_dim == 5
    assert len(default._grid.centroids) == 800
    assert evolve._centroids(6).shape == (800, 6)
    with pytest.raises(ValueError, match="axes"):
        evolve.merge([default, evolve.Archive(terrains="flat", axes="r1")])


def test_r1_run_saves_and_resumes_a_four_axis_archive(tmp_path):
    archive = evolve.run(n_init=2, n_batches=1, batch_size=2, budget_init=4,
                         budget_child=4, workers=1, episode_seconds=0.4,
                         variants=0, axes="r1", out_dir=str(tmp_path))
    assert archive.axes == "r1" and archive.cells
    path = tmp_path / "archive.pkl"
    loaded = evolve.load_archive(str(path))
    assert loaded.n_descriptors == 4 and loaded._grid.measure_dim == 4
    again = evolve.run(resume=str(path), n_batches=0, workers=1,
                       out_dir=str(tmp_path / "again"))
    assert again.n_descriptors == 4
    with pytest.raises(ValueError, match="axes"):
        evolve.run(resume=str(path), axes="body", n_batches=0, workers=1,
                   out_dir=str(tmp_path / "wrong"))


def test_train_accepts_the_r1_axes(tmp_path, monkeypatch):
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "train.py")
    spec = importlib.util.spec_from_file_location("train_axes_r1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    monkeypatch.setattr(mod.evolve, "run",
                        lambda **kw: calls.append(kw) or evolve.Archive(terrains="flat"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--init", "1", "--batches", "0",
                                      "--batch-size", "1", "--workers", "1",
                                      "--variants", "0", "--seconds", "1",
                                      "--out", str(tmp_path), "--axes", "r1"])
    mod.main()
    assert calls[-1]["axes"] == "r1"

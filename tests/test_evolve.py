"""
test_evolve.py
--------------
MAP-Elites cells, archive replacement and a tiny end to end run.
"""

import pickle

import numpy as np

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo


def _morph(thigh, shin, pairs=1, wheeled=False):
    legs = [mo.LegPair(i, 0.0, 0.0, thigh, shin, 0.02, wheeled, 0.05)
            for i in range(pairs)]
    return mo.Morphology(body_length=0.4, body_width=0.2, pairs=legs)


def test_cell_of_bins_leg_length():
    assert evolve.cell_of(_morph(0.12, 0.12)) == (1, 0, 1)
    assert evolve.cell_of(_morph(0.02, 0.02)) == (1, 0, 0)
    assert evolve.cell_of(_morph(0.25, 0.25)) == (1, 0, 4)
    assert evolve.cell_of(_morph(0.10, 0.10, pairs=2, wheeled=True)) == (2, 2, 1)


def test_insert_keeps_better_elite():
    genes = np.full(mo.MORPH_LENGTH, 0.5)
    ctrl = np.zeros(co.CTRL_LENGTH)
    cell = evolve.cell_of(mo.decode(genes))
    archive = evolve.Archive(terrains="flat")

    assert archive.insert(evolve.Elite(genes, ctrl, 1.0, [1.0], 0))
    assert not archive.insert(evolve.Elite(genes, ctrl, 0.5, [0.5], 0))
    assert archive.cells[cell].fitness == 1.0
    assert archive.insert(evolve.Elite(genes, ctrl, 2.0, [2.0], 1))
    assert archive.best().fitness == 2.0


def test_run_writes_archive(tmp_path):
    archive = evolve.run(n_init=2, n_batches=1, batch_size=2,
                         budget_init=6, budget_child=4, workers=1,
                         out_dir=str(tmp_path))

    path = tmp_path / "archive.pkl"
    assert path.exists()
    assert archive.cells
    assert len(archive.history) == 2
    with open(path, "rb") as f:
        assert isinstance(pickle.load(f), evolve.Archive)


def test_cell_name_round_trip():
    from darwinbot.evolve import cell_name, parse_cell_name
    for cell in [(1, 0, 0), (3, 0, 2), (2, 2, 4)]:
        assert parse_cell_name(cell_name(cell)) == cell
    assert cell_name((3, 0, 2)) == "6legs-0wheels-medium"

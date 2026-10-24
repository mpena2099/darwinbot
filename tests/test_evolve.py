"""
test_evolve.py
--------------
MAP-Elites cells, archive replacement and a tiny end to end run.
"""

import pickle

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo


def _morph(thigh, shin, pairs=1, wheeled=False):
    legs = [mo.LegPair(i, 0.0, 0.0, thigh, shin, False, wheeled, 0.05)
            for i in range(pairs)]
    return mo.Morphology(body_length=0.4, body_width=0.2, pairs=legs)


def test_cell_of_bins_servos():
    assert evolve.cell_of(_morph(0.06, 0.06)) == (1, 0, 0)            # 4 servos
    assert evolve.cell_of(_morph(0.10, 0.10, pairs=3)) == (3, 0, 4)   # 12
    assert evolve.cell_of(_morph(0.10, 0.10, pairs=2, wheeled=True)) == (2, 2, 4)


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
    for cell in [(1, 0, 0), (3, 1, 5), (2, 2, 10)]:
        assert parse_cell_name(cell_name(cell)) == cell
    assert cell_name((3, 1, 5)) == "6legs-2wheels-14servos"
    for bad in ("6legs-2wheels-medium", "6legs-2wheels-5servos",
                "6legs-2wheels-26servos"):
        with pytest.raises(ValueError):
            parse_cell_name(bad)


def _edge_genes(servos):
    """Three pairs; yaw and wheel switches chosen to hit a servo count."""
    genes = np.full(mo.MORPH_LENGTH, 0.5)
    extra = (servos - 12) // 2          # pairs beyond the 12 base servos
    for p in range(mo.N_PAIRS):
        b = 2 + p * mo.PAIR_GENES
        genes[b] = 1.0
        genes[b + 5] = 1.0 if p < extra else 0.0
        genes[b + 6] = 1.0 if p < extra - mo.N_PAIRS else 0.0
    return genes


def test_grid_cells_match_cell_of():
    rng = np.random.default_rng(7)
    genes = [rng.uniform(0.0, 1.0, mo.MORPH_LENGTH) for _ in range(8)]
    genes += [_edge_genes(n) for n in (12, 16, 24)]
    archive = evolve.Archive(terrains="flat")

    for i, g in enumerate(genes):
        elite = evolve.Elite(g, np.zeros(co.CTRL_LENGTH), float(i), [0.0], 0)
        assert archive.insert(elite)
        assert evolve.cell_of(mo.decode(g)) in archive.cells

    for g in genes:
        cell = evolve.cell_of(mo.decode(g))
        idx = int(archive._grid.index_of([mo.descriptor(mo.decode(g))])[0])
        grid = tuple(int(v) for v in archive._grid.int_to_grid_index([idx])[0])
        # The pair axis is centred on 1..3, so its ribs index is zero based.
        assert grid[1:] == cell[1:]
        assert grid[0] == cell[0] - 1


def test_old_archive_is_rejected():
    archive = evolve.Archive(terrains="flat")
    state = dict(archive.__dict__)
    state.pop("body_model")
    with pytest.raises(ValueError, match="older body model"):
        evolve.Archive.__new__(evolve.Archive).__setstate__(state)

"""
test_search_bias.py
-------------------
Opt-in mutation bias and tie replacement on fewer parts, both off by default.
"""

import pickle

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo


BEHAVIOUR = np.array([0.9, 0.3, 8, 0.0, 1.0, 2, 0.0], dtype=float)
CTRL = np.zeros(co.CTRL_LENGTH)


def _elite(fitness, genes):
    return evolve.Elite(genes, CTRL, fitness, [fitness], 0, BEHAVIOUR)


def _pair():
    """Two seeded bodies in the same cell, one with clearly fewer parts."""
    rng = np.random.default_rng(2)
    big, small = mo.random_morphology(rng), mo.random_morphology(rng)
    assert mo.decode(big).n_nodes > mo.decode(small).n_nodes
    return big, small


def _archive(tie_parts):
    return evolve.Archive(terrains="flat", search={"tie_parts": tie_parts})


def test_default_mutation_weights_unchanged():
    assert list(mo.mutation_weights("default")) == [p for _, p in mo.MUTATIONS]
    assert mo.MUTATIONS[2][1] == 0.2 and mo.MUTATIONS[3][1] == 0.1


def test_remove_bias_weights():
    weights = mo.mutation_weights("remove")
    assert np.allclose(weights, [0.30, 0.15, 0.10, 0.25, 0.15, 0.05, 0.05])
    for i in (0, 1, 4, 5, 6):            # every other operator keeps its weight
        assert weights[i] == mo.MUTATIONS[i][1]


def test_bias_reaches_mutate_and_workers(monkeypatch):
    mo.set_mutation_bias("remove")
    try:
        assert np.allclose(mo.mutation_weights(), mo.mutation_weights("remove"))
        with pytest.raises(ValueError):
            mo.set_mutation_bias("bogus")
        seen = []
        monkeypatch.setattr(mo, "mutate", lambda g, rng, **kw: seen.append(
            kw.get("mutation_bias")) or np.array(g))
        genes = np.full(mo.MORPH_LENGTH, 0.5)
        evolve.offspring(genes, genes, False, np.random.default_rng(0),
                         mutation_bias="remove")
        assert seen == ["remove"]
    finally:
        mo.set_mutation_bias("default")


def test_tie_replaces_fewer_parts_within_margin():
    big, small = _pair()
    a = _archive(0.02)
    assert a.insert(_elite(1.0, big))
    assert a.insert(_elite(0.99, small))            # 0.99 >= 1 - 0.02 * 1
    cell = evolve.cell_of(BEHAVIOUR)
    assert a.cells[cell].fitness == pytest.approx(0.99)
    assert np.allclose(a.cells[cell].morph, small)
    assert len(a.cells) == 1


def test_no_tie_with_eps_zero():
    big, small = _pair()
    a = _archive(0.0)
    a.insert(_elite(1.0, big))
    assert not a.insert(_elite(0.99, small))
    assert a.cells[evolve.cell_of(BEHAVIOUR)].fitness == 1.0


def test_no_tie_with_more_parts():
    big, small = _pair()
    a = _archive(0.02)
    a.insert(_elite(1.0, small))
    assert not a.insert(_elite(0.99, big))
    assert a.cells[evolve.cell_of(BEHAVIOUR)].fitness == 1.0


def test_no_tie_below_margin():
    big, small = _pair()
    a = _archive(0.02)
    a.insert(_elite(1.0, big))
    assert not a.insert(_elite(0.90, small))        # 0.90 < 1 - 0.02
    assert a.cells[evolve.cell_of(BEHAVIOUR)].fitness == 1.0


def test_better_child_still_replaces_with_tie_on():
    big, small = _pair()
    a = _archive(0.02)
    a.insert(_elite(1.0, big))
    assert a.insert(_elite(1.5, small))
    assert a.cells[evolve.cell_of(BEHAVIOUR)].fitness == 1.5


def test_merge_and_lineage_survive_tie_parts():
    big, small = _pair()
    a, b = _archive(0.02), _archive(0.02)
    a.insert(_elite(1.0, big), "pa")
    b.insert(_elite(0.99, small), "pb")
    out = evolve.merge([a, b])
    cell = evolve.cell_of(BEHAVIOUR)
    assert out.cells[cell].fitness == pytest.approx(0.99)
    assert out.lineage[evolve.elite_id(small, CTRL)][0] == "pb"
    assert evolve.elite_id(big, CTRL) in out.lineage


def test_options_survive_save_load(tmp_path):
    a = evolve.Archive(terrains="flat", search={"mutation_bias": "remove",
                                               "tie_parts": 0.02})
    path = tmp_path / "archive.pkl"
    with open(path, "wb") as f:
        pickle.dump(a, f)
    b = evolve.load_archive(str(path))
    assert b.search["mutation_bias"] == "remove"
    assert b.search["tie_parts"] == 0.02

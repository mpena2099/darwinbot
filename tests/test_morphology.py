"""
test_morphology.py
------------------
Genome decode, mutation and MAP-Elites descriptor.
"""

import numpy as np
import pytest

from darwinbot import morphology as mo

PAIR_ATTRS = {"attach_x": 1, "splay": 2, "thigh_len": 3, "shin_len": 4,
              "radius": 5, "wheel_rad": 7}


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_decode_stays_within_bounds(fill):
    m = mo.decode(np.full(mo.MORPH_LENGTH, fill))
    assert mo.BODY_BOUNDS[0][0] <= m.body_length <= mo.BODY_BOUNDS[0][1]
    assert mo.BODY_BOUNDS[1][0] <= m.body_width <= mo.BODY_BOUNDS[1][1]
    for pair in m.pairs:
        for attr, index in PAIR_ATTRS.items():
            lo, hi = mo.PAIR_BOUNDS[index]
            assert lo <= getattr(pair, attr) <= hi


@pytest.mark.parametrize("fill", [0.0, 0.2])
def test_no_enabled_gene_yields_one_pair(fill):
    m = mo.decode(np.full(mo.MORPH_LENGTH, fill))
    assert len(m.pairs) == 1
    assert m.n_legs == 2


def test_all_ones_enables_every_pair():
    m = mo.decode(np.full(mo.MORPH_LENGTH, 1.0))
    assert len(m.pairs) == mo.N_PAIRS
    assert m.n_wheels == 2 * mo.N_PAIRS


def test_mutate_keeps_shape_and_range():
    rng = np.random.default_rng(0)
    genes = mo.random_morphology(rng)
    child = mo.mutate(genes, rng)
    assert child.shape == genes.shape
    assert np.all(child >= 0.0) and np.all(child <= 1.0)


def test_descriptor():
    pairs = [
        mo.LegPair(0, 0.0, 0.0, 0.10, 0.10, 0.02, True, 0.05),
        mo.LegPair(1, 0.0, 0.0, 0.20, 0.20, 0.02, False, 0.05),
    ]
    m = mo.Morphology(body_length=0.4, body_width=0.2, pairs=pairs)
    n_pairs, wheeled, leg = mo.descriptor(m)
    assert (n_pairs, wheeled) == (2, 1)
    assert leg == pytest.approx(0.30)

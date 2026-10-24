"""
test_morphology.py
------------------
Genome decode, mutation and MAP-Elites descriptor.
"""

import numpy as np
import pytest

from darwinbot import morphology as mo

PAIR_ATTRS = {"attach_x": 1, "splay": 2, "wheel_rad": 7}


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
        mo.LegPair(0, 0.0, 0.0, 0.06, 0.06, True, True, 0.05),
        mo.LegPair(1, 0.0, 0.0, 0.12, 0.12, False, False, 0.05),
    ]
    m = mo.Morphology(body_length=0.4, body_width=0.2, pairs=pairs)
    assert mo.descriptor(m) == (2, 1, 12)
    assert m.mean_leg_length == pytest.approx(0.18)


def _genes(n_pairs, yaw=(), wheel=()):
    genes = np.zeros(mo.MORPH_LENGTH)
    for p in range(n_pairs):
        b = 2 + p * mo.PAIR_GENES
        genes[b] = 1.0
        genes[b + 5] = 1.0 if p in yaw else 0.0
        genes[b + 6] = 1.0 if p in wheel else 0.0
    return genes


@pytest.mark.parametrize("n_pairs, yaw, wheel, expected", [
    (1, (), (), 4),
    (1, (0,), (), 6),
    (1, (0,), (0,), 8),
    (2, (0,), (1,), 12),
    (3, (0, 1, 2), (0, 1, 2), 24),
])
def test_n_servos(n_pairs, yaw, wheel, expected):
    assert mo.decode(_genes(n_pairs, yaw, wheel)).n_servos == expected


def test_link_lengths_snap_to_catalogue():
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = mo.decode(mo.random_morphology(rng))
        for pair in m.pairs:
            assert pair.thigh_len in mo.LINK_LENGTHS
            assert pair.shin_len in mo.LINK_LENGTHS
            assert pair.radius == mo.LINK_RADIUS
    genes = _genes(1)
    genes[2 + 3], genes[2 + 4] = 0.0, 1.0
    pair = mo.decode(genes).pairs[0]
    assert (pair.thigh_len, pair.shin_len) == (mo.LINK_LENGTHS[0], mo.LINK_LENGTHS[-1])


def test_mutate_can_flip_has_yaw():
    rng = np.random.default_rng(0)
    genes = np.full(mo.MORPH_LENGTH, 0.9)
    flipped = [mo.mutate(genes, rng, sigma=0.0, flip_prob=1.0) for _ in range(2)]
    assert all(c[2 + 5] == pytest.approx(0.1) for c in flipped)

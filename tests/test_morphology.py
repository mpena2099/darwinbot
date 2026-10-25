"""
test_morphology.py
------------------
Tree genome: decode/encode, limits, mutation, mirroring and templates.
"""

import numpy as np
import pytest

from darwinbot import morphology as mo


def _same(a, b):
    assert (a.body_length, a.body_width) == pytest.approx((b.body_length, b.body_width))
    assert sorted(a.nodes) == sorted(b.nodes)
    for i, n in a.nodes.items():
        m = b.nodes[i]
        assert (n.kind, n.parent, n.slot, n.axis, n.flip) == \
            (m.kind, m.parent, m.slot, m.axis, m.flip)
        for field in ("rot", "length", "wheel_rad", "width", "thick",
                      "foot_rad", "tail_mass", "offset", "angle"):
            assert getattr(n, field) == pytest.approx(getattr(m, field))


def _within_limits(m):
    assert m.n_nodes <= mo.MAX_NODES
    assert m.n_servos <= mo.MAX_SERVOS
    assert m.max_depth <= mo.MAX_DEPTH
    assert m.n_servos >= 1
    slots = [(n.parent, n.slot) for n in m.nodes.values() if n.parent >= 0]
    assert len(slots) == len(set(slots))
    for n in m.nodes.values():
        if n.parent >= 0:
            assert n.slot < mo.N_SLOTS[m.nodes[n.parent].kind]
        assert 0.0 <= n.rot < 2 * np.pi
        assert 0.0 <= n.offset <= 1.0 and 0.0 <= n.angle <= 1.0
        if n.kind in ("joint", "link", "spring", "tail"):
            assert mo.LENGTH_RANGE[0] <= n.length <= mo.LENGTH_RANGE[1]
            assert mo.LINK_WIDTH_RANGE[0] <= n.width <= mo.LINK_WIDTH_RANGE[1]
        if n.kind == "spring":
            assert mo.SPRING_BLADE_RANGE[0] <= n.thick <= mo.SPRING_BLADE_RANGE[1]
        elif n.kind in ("joint", "link"):
            assert mo.LINK_THICK_RANGE[0] <= n.thick <= mo.LINK_THICK_RANGE[1]
        if n.kind == "tail":
            assert mo.TAIL_MASS_RANGE[0] <= n.tail_mass <= mo.TAIL_MASS_RANGE[1]
        if n.kind == "wheel":
            assert mo.WHEEL_RANGE_M[0] <= n.wheel_rad <= mo.WHEEL_RANGE_M[1]
        if n.kind == "foot":
            assert mo.FOOT_RANGE[0] <= n.foot_rad <= mo.FOOT_RANGE[1]


@pytest.mark.parametrize("fill", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_decode_any_vector_is_valid(fill):
    _within_limits(mo.decode(np.full(mo.MORPH_LENGTH, fill)))


def test_decode_random_vectors_are_valid():
    rng = np.random.default_rng(1)
    for _ in range(200):
        _within_limits(mo.decode(rng.uniform(0, 1, mo.MORPH_LENGTH)))


def test_encode_decode_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(100):
        genes = mo.random_morphology(rng)
        m = mo.decode(genes)
        _same(m, mo.decode(mo.encode(m)))
        assert np.allclose(mo.encode(m), genes)        # already canonical


def test_random_genes_decode_to_valid_bodies():
    rng = np.random.default_rng(9)
    for _ in range(200):
        m = mo.decode(rng.uniform(0, 1, mo.MORPH_LENGTH))
        _within_limits(m)
        _same(m, mo.decode(mo.encode(m)))


def test_repair_is_canonical():
    genes = np.random.default_rng(3).uniform(0, 1, mo.MORPH_LENGTH)
    fixed = mo.repair(genes)
    assert np.allclose(mo.repair(fixed), fixed)
    _same(mo.decode(genes), mo.decode(fixed))


def test_slot_ids_are_stable_block_indices():
    m = mo.template(1)
    genes = mo.encode(m)
    for i in m.nodes:
        assert genes[i * mo.BLOCK_GENES + mo.ACTIVE] == 1.0
    assert mo.decode(genes).nodes.keys() == m.nodes.keys()


def test_limits_hold_after_500_mutations():
    rng = np.random.default_rng(4)
    genes = mo.random_morphology(rng)
    for _ in range(500):
        genes = mo.mutate(genes, rng)
        assert genes.shape == (mo.MORPH_LENGTH,)
        assert np.all(genes >= 0.0) and np.all(genes <= 1.0)
        _within_limits(mo.decode(genes))


def test_mutation_from_a_body_at_the_servo_cap_respects_limits():
    rng = np.random.default_rng(5)
    genes = mo.encode(mo.template(2, wheels=True, yaw=True))
    assert mo.decode(genes).n_servos == mo.MAX_SERVOS
    for _ in range(100):
        _within_limits(mo.decode(mo.mutate(genes, rng, n_ops=3)))


def _assert_mirrored(m, a, b):
    """Subtree b is the reflection of subtree a across the XZ plane."""
    na, nb = m.nodes[a], m.nodes[b]
    assert (na.kind, na.axis, na.wheel_rad, na.foot_rad) == \
        (nb.kind, nb.axis, nb.wheel_rad, nb.foot_rad)
    for field in ("length", "width", "thick", "offset"):
        assert getattr(na, field) == pytest.approx(getattr(nb, field))
    if na.parent == 0:                    # yaw mirrors around the torso
        assert nb.angle == pytest.approx(1.0 - na.angle)
    else:
        assert nb.angle == pytest.approx(na.angle)
    assert nb.rot == pytest.approx((-na.rot) % (2 * np.pi))
    assert nb.flip == (na.flip if na.axis == "yaw" or na.kind != "joint" else not na.flip)
    ca, cb = m.child_map().get(a, {}), m.child_map().get(b, {})
    assert {mo.mirror_slot(na.kind, s) for s in ca} == set(cb)
    for s, c in ca.items():
        _assert_mirrored(m, c, cb[mo.mirror_slot(na.kind, s)])


def test_mirrored_duplication_is_symmetric():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "joint", 0, 2, 3 * np.pi / 2, "pitch", 0.10,
                         flip=True, offset=0.3, angle=0.3)
    m.nodes[2] = mo.Node(2, "joint", 1, 1, np.pi / 2, "roll", 0.08)
    m.nodes[3] = mo.Node(3, "wheel", 2, 0, np.pi, wheel_rad=0.04)
    m.nodes[4] = mo.Node(4, "foot", 1, 2, np.pi)
    twin = mo.copy_subtree(m, 1, 0, mo.mirror_slot("core", 2), mirror=True)
    assert m.nodes[twin].slot == 5
    _assert_mirrored(m, 1, twin)

    # geometry: every module of the copy is the reflection of its original
    fr = m.frames()
    left = [i for i in m.order if i in (1, 2, 3, 4)]
    right = [i for i in m.order if i > 4]
    assert len(right) == len(left) == 4
    by_kind = lambda ids: sorted((m.nodes[i].kind, tuple(np.round(fr[i][0] * [1, -1, 1], 6)))
                                 for i in ids)
    assert by_kind(left) == sorted((m.nodes[i].kind, tuple(np.round(fr[i][0], 6)))
                                   for i in right)


def test_mutation_mirror_operator_makes_symmetric_pairs():
    rng = np.random.default_rng(8)
    m = mo.template(1)
    cm = m.child_map()[0]
    assert set(cm) == {3, 6}                      # left and right middle slots
    _assert_mirrored(m, cm[3], cm[6])
    # drop the right leg, let the operator rebuild it
    keep = {i for i in m.nodes if i == 0 or _under(m, i, cm[3])}
    m.nodes = {i: n for i, n in m.nodes.items() if i in keep}
    for _ in range(100):          # the operator picks any mirrorable subtree
        mo._op_mirror(m, rng)
        if 6 in m.child_map()[0]:
            break
    cm = m.child_map()[0]
    assert set(cm) == {3, 6}
    _assert_mirrored(m, cm[3], cm[6])


def _under(m, i, ancestor):
    while i >= 0:
        if i == ancestor:
            return True
        i = m.nodes[i].parent if i in m.nodes else -1
    return False


def test_template_right_leg_mirrors_left():
    m = mo.template(2, wheels=True, yaw=True)
    cm = m.child_map()[0]
    for left, right in ((2, 5), (4, 7)):
        _assert_mirrored(m, cm[left], cm[right])


def test_template_counts():
    # (pairs asked, yaw, wheels, pairs built, servos); a pair over the cap is left out
    for asked, yaw, wheels, built, servos in [
            (1, False, False, 1, 4), (1, True, False, 1, 6), (1, True, True, 1, 8),
            (2, True, True, 2, 16), (3, False, False, 3, 12), (3, True, False, 2, 12)]:
        m = mo.template(asked, wheels, yaw)
        assert m.n_servos == servos
        assert m.n_wheels == (2 * built if wheels else 0)
        assert m.n_terminals == 2 * built
        _within_limits(m)


def test_template_layout_follows_model_order():
    m = mo.template(2, wheels=True, yaw=True)
    layout = m.layout()
    assert len(layout) == m.n_servos == 16
    assert [a.slot for a in layout] == [
        i for i in m.order if m.nodes[i].kind in ("joint", "wheel")]
    assert layout[0] == mo.Actuator(1, "joint", "yaw")
    assert sum(a.kind == "wheel" for a in layout) == 4


def test_random_population_mixes_seeds_and_trees():
    rng = np.random.default_rng(6)
    legs = other = 0
    for _ in range(60):
        m = mo.decode(mo.random_morphology(rng))
        pure = all(n.kind in ("core", "joint", "wheel") and n.axis in ("pitch", "yaw")
                   for n in m.nodes.values())
        legs += pure
        other += not pure
    assert legs > 10 and other > 10


def test_n_servos_counts_joints_and_wheels_only():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "link", 0, 0)
    m.nodes[2] = mo.Node(2, "joint", 1, 0)
    m.nodes[3] = mo.Node(3, "foot", 2, 0)
    m.nodes[4] = mo.Node(4, "wheel", 1, 1)
    assert (m.n_joints, m.n_wheels, m.n_servos) == (1, 1, 2)
    assert m.n_nodes == 5


def test_brick_gets_a_servo():
    assert mo.decode(np.zeros(mo.MORPH_LENGTH)).n_servos == 1


def test_terminal_points_and_reach():
    m = mo.template(2, wheels=False)
    pts = m.terminal_points()
    assert len(pts) == 4
    for i, p, rad in pts:
        assert p[2] < 0 and rad == mo.section_radius(m.nodes[i])   # shins hang down
    assert m.max_reach >= max(np.linalg.norm(p) for _, p, _ in pts)
    assert m.lowest_z() == pytest.approx(min(p[2] - r for _, p, r in pts), abs=0.03)


def test_prune_drops_deepest_first():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    for i in range(1, 8):
        m.nodes[i] = mo.Node(i, "link", i - 1, 0)
    m.nodes[8] = mo.Node(8, "joint", 0, 3, 3 * np.pi / 2)
    mo.prune(m)
    assert m.max_depth <= mo.MAX_DEPTH
    assert 5 in m.nodes and 6 not in m.nodes


def test_summary_mentions_every_module():
    m = mo.template(1, wheels=True)
    text = mo.summary(m)
    assert text.count("#") == m.n_nodes
    assert "wheel" in text and "\u2014" not in text


def test_elbow_offset_and_angle_reproduce_todays_frame():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "joint", 0, 3, 0.0, "pitch", 0.10)
    m.nodes[2] = mo.Node(2, "joint", 1, 1, 0.0, "pitch", 0.08,
                         offset=mo.ELBOW_OFFSET)
    for slot, side in ((1, 1.0), (2, -1.0)):
        off, rot = m.slot_frame(m.nodes[1], slot, m.nodes[2])
        assert off == pytest.approx([0.10, side * 0.04, 0.0])
        assert rot == pytest.approx(mo._rz(side * np.pi / 2))


def _single_link(width, thick, length=0.10):
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "joint", 0, 3, 0.0, "pitch", length,
                         width=width, thick=thick)
    return m


def test_thin_link_violates_strength():
    assert mo.violations(_single_link(0.012, 0.003))


def test_wide_link_passes_printability():
    assert mo.violations(_single_link(0.020, 0.006)) == []


def test_link_longer_than_the_bed_is_a_violation():
    assert mo.violations(_single_link(0.030, 0.006, length=0.20))


def _single_spring(blade=0.004):
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "spring", 0, 3, 0.0, axis="yaw", length=0.10,
                         width=0.020, thick=blade)
    return m


def test_spring_round_trip():
    m = _single_spring()
    m.nodes[2] = mo.Node(2, "joint", 1, 0, 0.0, "pitch", 0.08)
    genes = mo.encode(m)
    _same(m, mo.decode(genes))
    assert np.allclose(mo.encode(mo.decode(genes)), genes)


def test_spring_is_a_passive_link():
    m = _single_spring()
    assert (m.n_springs, m.n_servos, m.n_terminals) == (1, 0, 1)
    assert mo.N_SLOTS["spring"] == 3
    assert mo.spring_stiffness(m.nodes[1]) > 0
    assert not mo.violations(_single_spring(0.002))   # the PLA segment keeps its own thickness


def test_spring_stiffness_grows_with_the_cube_of_the_blade():
    k = lambda blade: mo.spring_stiffness(_single_spring(blade).nodes[1])
    assert k(0.004) == pytest.approx(8 * k(0.002))
    assert k(0.006) == pytest.approx(27 * k(0.002))
    assert k(0.008) == pytest.approx(64 * k(0.002))


def test_soft_spring_blade_does_not_fail_the_strength_check():
    m = mo.template(2)
    i = max(m.nodes) + 1
    m.nodes[i] = mo.Node(i, "spring", 0, 0, 0.0, length=0.08, width=0.02, thick=0.002)
    assert not mo.violations(m)
    assert mo.plate_thick(m.nodes[i]) == mo.SPRING_SEGMENT_THICK


TERMINALS = ("tail", "skid", "spike")


def _terminal(kind, length=0.12, tail_mass=0.05):
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, kind, 0, 0, 0.0,
                         length=length if kind == "tail" else 0.08,
                         tail_mass=tail_mass if kind == "tail" else 0.0)
    m.nodes[2] = mo.Node(2, "joint", 0, 1, 3 * np.pi / 2, "pitch", 0.10)
    return m


@pytest.mark.parametrize("kind", TERMINALS)
def test_terminal_round_trip(kind):
    m = _terminal(kind)
    genes = mo.encode(m)
    _same(m, mo.decode(genes))
    assert np.allclose(mo.encode(mo.decode(genes)), genes)


@pytest.mark.parametrize("kind", TERMINALS)
def test_terminal_helpers_include_the_module(kind):
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "joint", 0, 3, 3 * np.pi / 2, "pitch", 0.10)
    m.nodes[2] = mo.Node(2, kind, 1, 1, 0.0, length=0.12,
                         tail_mass=0.05 if kind == "tail" else 0.0)
    point = next((p, r) for j, p, r in m.terminal_points() if j == 2)
    p, rad = point
    assert m.max_reach >= np.linalg.norm(p) + rad
    low = {"tail": mo.TAIL_TIP_RADIUS, "skid": mo.SKID_SIZE[2] / 2,
           "spike": mo.SPIKE_RADIUS}[kind]
    assert m.lowest_z() == pytest.approx(p[2] - low)


def test_tail_mass_comes_from_the_wheel_gene():
    light = mo.decode(mo.encode(_terminal("tail", 0.10, 0.0)))
    heavy = mo.decode(mo.encode(_terminal("tail", 0.10, 0.10)))
    assert light.nodes[1].tail_mass == pytest.approx(0.0)
    assert heavy.nodes[1].tail_mass == pytest.approx(0.10)
    assert light.nodes[1].length == heavy.nodes[1].length == pytest.approx(0.10)

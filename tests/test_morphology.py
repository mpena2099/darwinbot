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
    assert a.contour == pytest.approx(b.contour)
    assert sorted(a.nodes) == sorted(b.nodes)
    for i, n in a.nodes.items():
        m = b.nodes[i]
        assert (n.kind, n.parent, n.slot, n.axis, n.flip) == \
            (m.kind, m.parent, m.slot, m.axis, m.flip)
        for field in ("rot", "length", "wheel_rad", "width", "thick",
                      "foot_rad", "tail_mass", "offset", "angle"):
            assert getattr(n, field) == pytest.approx(getattr(m, field))


def _within_limits(m):
    assert len(m.contour) == 8
    assert all(mo.CONTOUR_RANGE[0] <= c <= mo.CONTOUR_RANGE[1] for c in m.contour)
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


def test_contour_round_trip():
    m = mo.decode(mo.encode(mo.template(1)))
    m.contour = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.1, 0.9)
    genes = mo.encode(m)
    _same(m, mo.decode(genes))
    assert np.allclose(mo.encode(mo.decode(genes)), genes)


def test_ones_contour_is_the_old_rectangle():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    assert np.allclose(m.torso_polygon(), [
        (0.2, 0.0), (0.2, 0.1), (0.0, 0.1), (-0.2, 0.1),
        (-0.2, 0.0), (-0.2, -0.1), (0.0, -0.1), (0.2, -0.1)])


def test_contour_too_small_for_the_electronics_is_a_violation():
    tiny = mo.Morphology(0.15, 0.10, {0: mo.Node(0, "core")}, (0.4,) * 8)
    assert "torso contour does not hold the electronics" in mo.violations(tiny)
    assert "torso contour does not hold the electronics" not in mo.violations(
        mo.template(1))


def test_mirrored_core_mount_is_a_reflection_on_a_star_contour():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")},
                      (0.5, 1.3) * 4)
    m.nodes[1] = mo.Node(1, "joint", 0, 2, 0.0, "pitch", 0.10,
                         offset=0.3, angle=0.3)
    twin = mo.copy_subtree(m, 1, 0, mo.mirror_slot("core", 2), mirror=True)
    left, right = m.frames()[1][0], m.frames()[twin][0]
    assert right == pytest.approx([left[0], -left[1], left[2]])


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
    for field in ("length", "width", "thick"):
        assert getattr(na, field) == pytest.approx(getattr(nb, field))
    if na.parent == 0:                    # position and yaw mirror around the torso
        assert nb.offset == pytest.approx(1.0 - na.offset)
        assert nb.angle == pytest.approx(1.0 - na.angle)
    else:
        assert nb.offset == pytest.approx(na.offset)
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


def test_n_legs_counts_torso_limbs_with_a_joint_and_no_wheel():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "link", 0, 0)        # leg: link, joint, foot
    m.nodes[2] = mo.Node(2, "joint", 1, 0)
    m.nodes[3] = mo.Node(3, "foot", 2, 0)
    m.nodes[4] = mo.Node(4, "joint", 0, 1)       # wheeled limb: not a leg
    m.nodes[5] = mo.Node(5, "wheel", 4, 0)
    m.nodes[6] = mo.Node(6, "tail", 0, 2)        # passive, no joint: not a leg
    m.nodes[7] = mo.Node(7, "joint", 0, 3)       # bare joint: a leg
    assert m.n_legs == 2
    assert mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")}).n_legs == 0


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
        assert off == pytest.approx([0.10, side * 0.05, 0.0])
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


def test_bed_check_uses_the_path_bounding_box():
    straight = _single_link(0.030, 0.006, length=0.20)
    assert any("bed" in v for v in mo.violations(straight))
    for profile, bend in (("L", np.pi / 2), ("arc", -np.pi / 2)):
        bent = _single_link(0.030, 0.006, length=0.20)
        bent.nodes[1].profile, bent.nodes[1].bend = profile, bend
        assert not mo.violations(bent)


def test_profile_and_foot_shape_round_trip():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "joint", 0, 3, 0.0, "pitch", 0.10,
                         profile="L", bend=np.radians(30))
    m.nodes[2] = mo.Node(2, "link", 1, 0, 0.0, length=0.08,
                         profile="arc", bend=-np.radians(45))
    m.nodes[3] = mo.Node(3, "foot", 2, 0, 0.0, foot_rad=0.02,
                         foot_shape="claw")
    m.nodes[4] = mo.Node(4, "foot", 0, 2, 0.0, foot_rad=0.02,
                         foot_shape="pad")
    genes = mo.encode(m)
    m2 = mo.decode(genes)
    for i in (1, 2):
        assert m2.nodes[i].profile == m.nodes[i].profile
        assert m2.nodes[i].bend == pytest.approx(m.nodes[i].bend)
    assert m2.nodes[3].foot_shape == "claw"
    assert m2.nodes[4].foot_shape == "pad"
    assert np.allclose(mo.encode(m2), genes)
    # the profile/bend genes are ignored by a foot
    genes[3 * mo.BLOCK_GENES + mo.PROFILE] = 1.0
    genes[3 * mo.BLOCK_GENES + mo.BEND] = 1.0
    assert mo.decode(genes).nodes[3].profile == "straight"
    assert mo.decode(genes).nodes[3].bend == 0.0
    # the AXIS gene picks the foot shape
    genes[3 * mo.BLOCK_GENES + mo.AXIS] = 0.0
    assert mo.decode(genes).nodes[3].foot_shape == mo.FOOT_SHAPES[0]


@pytest.mark.parametrize("profile,bend", (("L", np.pi / 2),
                                          ("arc", np.pi / 2)))
def test_link_tip_for_bent_profiles(profile, bend):
    node = mo.Node(1, "link", length=0.12, profile=profile, bend=bend)
    path = mo.link_path(node)
    assert sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])) == \
        pytest.approx(node.length)
    pos, rot = mo.link_tip(node)
    assert rot == pytest.approx(mo._rz(np.pi / 2))
    if profile == "L":                      # two 60 mm pieces, the second up
        assert pos == pytest.approx([0.06, 0.06, 0.0])
    else:                                   # three 40 mm pieces, 45 deg each
        assert pos == pytest.approx([0.04 + 0.04 / np.sqrt(2),
                                     0.04 / np.sqrt(2) + 0.04, 0.0])


def test_mirror_negates_bend():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "link", 0, 2, 0.0, length=0.10,
                         profile="L", bend=np.radians(60))
    twin = mo.copy_subtree(m, 1, 0, mo.mirror_slot("core", 2), mirror=True)
    assert m.nodes[twin].profile == "L"
    assert m.nodes[twin].bend == pytest.approx(-np.radians(60))


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


def test_spring_is_a_passive_link(monkeypatch):
    # springs are disabled for training (no printed part yet); the simulator
    # still models them
    monkeypatch.setattr(mo, "DISABLED_KINDS", ())
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


def test_soft_spring_blade_does_not_fail_the_strength_check(monkeypatch):
    # springs are disabled for training (no printed part yet); the simulator
    # still models them
    monkeypatch.setattr(mo, "DISABLED_KINDS", ())
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


def test_springs_always_bend_about_local_y():
    """Spring stiffness models the blade's thin axis, so the AXIS gene and the
    axis mutation never apply to springs."""
    rng = np.random.default_rng(3)
    for _ in range(60):
        m = mo.decode(mo.mutate(mo.random_morphology(rng), rng))
        assert all(n.axis == "yaw" for n in m.nodes.values() if n.kind == "spring")
    genes = mo.encode(_single_spring())
    genes[mo.BLOCK_GENES + mo.AXIS] = 0.0          # would be pitch for a joint
    assert mo.decode(genes).nodes[1].axis == "yaw"
    n = mo._random_node(rng, "spring", 0, 3)
    assert n.axis == "yaw"


def test_ensure_servo_with_every_id_used():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    for i in range(1, mo.MAX_NODES):      # a chain of passive links
        m.nodes[i] = mo.Node(i, "link", i - 1, 0 if i > 1 else 3, 0.0)
    mo._ensure_servo(m)
    assert m.n_servos == 1 and m.n_nodes <= mo.MAX_NODES


def test_ensure_servo_with_every_core_slot_taken():
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    for s in range(8):
        m.nodes[s + 1] = mo.Node(s + 1, "link", 0, s, 0.0)
    m.nodes[9] = mo.Node(9, "link", 1, 0, 0.0)    # a child under the dropped one
    mo._ensure_servo(m)
    assert m.n_servos == 1
    slots = [(n.parent, n.slot) for n in m.nodes.values() if n.parent >= 0]
    assert len(slots) == len(set(slots))
    assert all(n.parent in m.nodes for n in m.nodes.values() if n.id)


def test_graft_copies_a_donor_limb_into_a_valid_body():
    rng = np.random.default_rng(0)
    host, donor = mo.encode(mo.template(1)), mo.encode(mo.template(2, wheels=True))
    grown = 0
    for _ in range(30):
        child = mo.graft(host, donor, rng)
        m = mo.decode(child)
        assert m.n_servos <= mo.MAX_SERVOS and m.n_nodes <= mo.MAX_NODES
        assert np.array_equal(mo.encode(m), child)          # already repaired
        grown += m.n_nodes > mo.template(1).n_nodes
        assert m.n_wheels <= mo.template(2, wheels=True).n_wheels
    assert grown > 15


def test_graft_without_room_returns_the_host():
    rng = np.random.default_rng(1)
    full = mo.encode(mo.template(3, wheels=True, yaw=True))
    lone = mo.encode(mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")}))
    assert mo.decode(mo.graft(full, lone, rng)).n_nodes == mo.decode(full).n_nodes


def test_scale_stretches_a_limb_within_range():
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(40):
        m = mo.template(2)
        mo._op_scale(m, rng)
        for n in m.nodes.values():
            if n.kind == "joint":
                assert mo.LENGTH_RANGE[0] <= n.length <= mo.LENGTH_RANGE[1]
                seen.add(round(n.length, 3))
    assert max(seen) > 0.15 and min(seen) < 0.07


def _spine_body(n_segments=1):
    """Template legs on the core plus a spine segment (and a leg on it)."""
    m = mo.template(2)
    first = max(m.nodes) + 1
    m.nodes[first] = mo.Node(first, "segment", 0, 1, 0, axis="pitch",
                             length=0.12, width=0.10)
    m.nodes[first + 1] = mo.Node(first + 1, "joint", first, 3, 0, axis="pitch",
                                 length=0.08)
    m.nodes[first + 2] = mo.Node(first + 2, "joint", first, 6, 0, axis="pitch",
                                 length=0.08)
    if n_segments > 1:
        m.nodes[first + 3] = mo.Node(first + 3, "segment", first, 0, 0,
                                     axis="roll", length=0.10, width=0.10)
    return m


def test_segment_roundtrip_and_counts():
    m = _spine_body()
    seg = next(i for i, n in m.nodes.items() if n.kind == "segment")
    back = mo.decode(mo.encode(m))
    _same(back, mo.decode(mo.encode(back)))
    assert sorted(back.nodes) == sorted(m.nodes)
    assert back.n_segments == 1
    assert back.n_servos == m.n_joints + m.n_wheels
    assert back.n_joints == sum(n.kind in ("joint", "segment")
                                for n in back.nodes.values())
    # legs on the core (template) plus two on the segment; the segment is no limb
    assert back.n_legs == mo.template(2).n_legs + 2
    acts = [a for a in back.layout() if a.slot == seg]
    assert acts == [mo.Actuator(seg, "joint", "pitch")]
    assert [a.slot for a in back.layout()] == [i for i in back.order
                                               if mo.is_joint(back.nodes[i].kind)
                                               or back.nodes[i].kind == "wheel"]
    assert mo.violations(back) == []


def test_decode_drops_misplaced_and_extra_segments():
    m = _spine_body(2)
    assert mo.decode(mo.encode(m)).n_segments == 2
    third = max(m.nodes) + 1
    m.nodes[third] = mo.Node(third, "segment", max(m.nodes) - 1, 1, 0)
    m.nodes[third + 1] = mo.Node(third + 1, "segment", 0, 2, 0)   # side slot
    d = mo.decode(mo.encode(m))
    assert d.n_segments == 2                 # both misplaced ones are dropped
    assert all(n.slot in (0, 1) for n in d.nodes.values() if n.kind == "segment")
    # a child on slot 1 of a segment is dropped
    seg = next(i for i, n in m.nodes.items() if n.kind == "segment")
    k = max(m.nodes) + 1
    m.nodes[k] = mo.Node(k, "joint", seg, 1, 0)
    d = mo.decode(mo.encode(m))
    assert not any(n.parent == seg and n.slot == 1 for n in d.nodes.values())
    assert (seg, 1) not in _spine_body().free_slots()


@pytest.mark.parametrize("seed", range(6))
def test_nothing_hangs_on_segment_slot_one(seed):
    rng = np.random.default_rng(seed)
    genes = mo.encode(_spine_body(2))
    for _ in range(150):
        genes = mo.mutate(genes, rng)
        m = mo.decode(genes)
        _within_limits(m)
        assert m.n_segments <= mo.MAX_SEGMENTS
        for n in m.nodes.values():
            if n.kind == "segment":
                assert n.parent >= 0 and n.slot in (0, 1)
                assert m.nodes[n.parent].kind in ("core", "segment")
            if n.parent >= 0 and m.nodes[n.parent].kind == "segment":
                assert n.slot != 1
        assert all(not (m.nodes[i].kind == "segment" and s == 1)
                   for i, s in m.free_slots())


def _face_body(slots=(8, 9, 10, 11)):
    m = mo.Morphology(0.30, 0.20, {0: mo.Node(0, "core")})
    for k, slot in enumerate(slots, start=1):
        m.nodes[k] = mo.Node(k, "joint", 0, slot, 0.0, angle=0.25 + 0.1 * k,
                             offset=0.3 + 0.1 * k)
    return m


def test_face_slots_round_trip():
    m = _face_body()
    _same(m, mo.decode(mo.encode(m)))
    assert mo.mirror_slot("core", 9) == 9
    assert mo.N_SLOTS["segment"] == 8


def test_face_slot_frame_points_up_and_down():
    m = _face_body()
    for k, slot in enumerate((8, 9, 10, 11), start=1):
        n = m.nodes[k]
        off, rot = m.slot_frame(m.nodes[0], slot, n)
        up = 1.0 if slot < 10 else -1.0
        assert off[2] == pytest.approx(up * mo.TORSO_HEIGHT / 2)
        assert rot[:, 0] == pytest.approx([0, 0, up])
        assert np.allclose(rot @ rot.T, np.eye(3)) and np.linalg.det(rot) > 0
        assert abs(rot[2, 1]) < 1e-9 and abs(rot[2, 2]) < 1e-9   # Y and Z horizontal
        yaw = (2 * n.angle - 1) * np.pi
        assert rot[:, 1] == pytest.approx([-np.sin(yaw), np.cos(yaw), 0])
        assert off[0] == pytest.approx(0.5 * m.torso_polygon()[0 if slot % 2 == 0 else 4][0])
        assert abs(off[1]) <= mo.FACE_Y * m.body_width / 2 + 1e-9


def test_servo_on_a_torso_face_is_not_buildable():
    short = mo.Morphology(0.15, 0.20, {0: mo.Node(0, "core"),
                                       1: mo.Node(1, "joint", 0, 10, 0.0)})
    assert any("torso face" in v for v in mo.violations(short))
    passive = mo.Morphology(0.15, 0.20, {0: mo.Node(0, "core"),
                                         1: mo.Node(1, "foot", 0, 10, 0.0)})
    assert not mo.violations(passive)


def _long_spine(n=None):
    """A core with a chain of MAX_SEGMENTS spine plates and a leg pair on the last."""
    n = mo.MAX_SEGMENTS if n is None else n
    m = mo.template(1)
    first = max(m.nodes) + 1
    prev, slot = 0, 1
    for k in range(n):
        m.nodes[first + k] = mo.Node(first + k, "segment", prev, slot, 0,
                                     axis="yaw" if k % 2 else "pitch",
                                     length=0.10, width=0.10)
        prev, slot = first + k, 0
    for j, s in enumerate((3, 6)):
        i = first + n + j
        m.nodes[i] = mo.Node(i, "joint", prev, s, 0, axis="pitch", length=0.08)
    return m, prev


def test_six_segment_spine_decodes_and_segments_add_no_depth():
    assert mo.MAX_SEGMENTS == 6
    m, last = _long_spine()
    d = mo.decode(mo.encode(m))
    assert d.n_segments == 6 and sorted(d.nodes) == sorted(m.nodes)
    assert d.depth(last) == 0 and d.max_depth == mo.template(1).max_depth
    assert mo.violations(d) == []
    # a seventh plate is dropped, the limbs stay
    m7, _ = _long_spine(7)
    d7 = mo.decode(mo.encode(m7))
    assert d7.n_segments == 6
    # limbs still stop at MAX_DEPTH behind the spine
    k = max(m.nodes) + 1
    parent = max(i for i, n in m.nodes.items() if n.kind == "joint")
    for j in range(mo.MAX_DEPTH + 1):
        m.nodes[k + j] = mo.Node(k + j, "link", parent, 0, 0, length=0.08)
        parent = k + j
    assert mo.decode(mo.encode(m)).max_depth <= mo.MAX_DEPTH


def test_long_spine_is_buildable():
    from darwinbot.sim import Episode
    import darwinbot.terrain as te
    m, _ = _long_spine()
    d = mo.decode(mo.encode(m))
    assert Episode(d, te.flat(), seconds=0.3).buildable


@pytest.mark.parametrize("seed", range(4))
def test_operators_respect_the_longer_spine(seed):
    rng = np.random.default_rng(seed)
    g = mo.encode(_long_spine()[0])
    for _ in range(30):
        g = mo.mutate(g, rng)
        m = mo.decode(g)
        assert m.n_segments <= mo.MAX_SEGMENTS and m.max_depth <= mo.MAX_DEPTH
        assert m.n_nodes <= mo.MAX_NODES and m.n_servos <= mo.MAX_SERVOS


def test_wheel_width_gene_roundtrip_and_range():
    m = mo.template(1, wheels=True)
    w = next(i for i, n in m.nodes.items() if n.kind == "wheel")
    for width in (0.010, 0.035, 0.060):
        m.nodes[w].wheel_width = width
        back = mo.decode(mo.encode(m))
        assert back.nodes[w].wheel_width == pytest.approx(width, abs=1e-6)
    g = mo.encode(m)
    assert g.size == mo.MAX_NODES * mo.BLOCK_GENES


def test_fit_clashes_turns_a_torso_wheel_instead_of_dropping_it():
    from darwinbot import builder
    m = mo.Morphology(0.32, 0.18, {0: mo.Node(0, "core")})
    for node in (mo.Node(1, "wheel", 0, 2, angle=0.5, wheel_rad=0.06, wheel_width=0.02),
                 mo.Node(2, "wheel", 0, 5, angle=0.5, wheel_rad=0.06, wheel_width=0.02),
                 mo.Node(3, "caster", 0, 1, wheel_rad=0.025)):
        m.nodes[node.id] = node
    assert builder.spawn_clashes(m)                  # both wheels cut into the plate
    mo.fit_clashes(m)
    assert {1, 2} <= set(m.nodes) and m.nodes[1].kind == m.nodes[2].kind == "wheel"
    assert builder.spawn_clashes(m) == []


def test_random_trees_put_wheels_straight_on_the_torso():
    bodies = [mo.decode(mo.random_morphology(np.random.default_rng(seed)))
              for seed in range(200)]
    cars = [m for m in bodies if any(
        n.kind == "wheel" and m.nodes[n.parent].kind == "core"
        for n in m.nodes.values() if n.parent >= 0)]
    assert len(cars) >= 10                           # 5% of the bodies

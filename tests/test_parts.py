"""Tests for the parametric printable parts."""

import pytest

pytest.importorskip("build123d")

from build123d import Box, Cylinder, Pos, Rot  # noqa: E402

from darwinbot import parts  # noqa: E402

HIPS = [(50, 40), (50, -40), (-50, 40), (-50, -40)]


def _polygon(length, width):
    return [(length / 2, 0), (length / 2, width / 2), (0, width / 2),
            (-length / 2, width / 2), (-length / 2, 0),
            (-length / 2, -width / 2), (0, -width / 2), (length / 2, -width / 2)]


def _star(length=400, width=200):
    poly = _polygon(length, width)
    return [(x * (1.3 if k % 2 else 0.5), y * (1.3 if k % 2 else 0.5))
            for k, (x, y) in enumerate(poly)]


def _ok(p):
    assert p.is_valid
    assert p.volume > 0


def _cut(part, solid):
    """Volume of `part` inside `solid`."""
    return (part & solid).volume if (part & solid).solids() else 0.0


def _hole_is_free(part, x, y, z, r=1.4, axis="Z"):
    """True when a thin pin of radius r fits through the part at (x, y, z)."""
    pin = Cylinder(r, 100)
    if axis == "X":
        pin = Rot(Y=90) * pin
    return _cut(part, Pos(x, y, z) * pin) < 1e-6


def _mount_points(x0=0.0, y0=0.0, yaw=0.0):
    import math
    h = parts.MOUNT_SIDE / 2
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return [(x0 + c * sx - s * sy, y0 + s * sx + c * sy)
            for sx in (-h, h) for sy in (-h, h)]


def test_servo_bracket():
    p = parts.servo_bracket()
    _ok(p)
    for x, y in _mount_points():
        assert _hole_is_free(p, x, y, 1.0)
    # the origin is the output axis: the body centre sits SERVO_AXIS_X behind
    c = -parts.SERVO_AXIS_X
    # four M2 holes up into the body's tapped holes (bottom face pattern)
    for x in (5.0, -19.5):
        for y in (-10.2, 10.2):
            assert _hole_is_free(p, c + x, y, 1.0, r=1.0)
    # no ears: the floor is solid where the old ear screws went
    for x in (-20.0, 20.0):
        assert not _hole_is_free(p, c + x, 0.0, 1.0)
    bb = p.bounding_box()
    assert abs((bb.min.X + bb.max.X) / 2 - c) < 0.01


def test_servo_axis_offset_matches_the_simulation():
    from darwinbot import morphology as mo
    assert abs(mo.SERVO_AXIS_OFFSET * 1000 - parts.SERVO_AXIS_X) < 1e-9
    assert mo.SERVO_CENTRE[0] < 0


def test_horn_holes_follow_the_real_horn():
    # 4 holes on a 9.9 mm square at 45 degrees to the body axes
    assert parts.HORN_N_HOLES == 4
    assert parts.HORN_BOLT_CIRCLE == pytest.approx(14.0)
    x, y = zip(*parts.horn_hole_centres())
    assert max(map(abs, x)) == pytest.approx(4.95, abs=0.02)
    assert max(map(abs, y)) == pytest.approx(4.95, abs=0.02)
    p = parts.link(80, 20, 6)
    for px, py in parts.horn_hole_centres():
        assert _hole_is_free(p, px, py, 3.0, r=1.0)


def test_mount_pattern_is_a_square_that_fits_the_boss():
    import math
    assert parts.MOUNT_SIDE == 16.0
    corner = parts.MOUNT_SIDE / math.sqrt(2) + parts.M3 / 2
    assert corner + 1.0 <= parts.END_BOSS_D / 2      # at least 1 mm of rim


@pytest.mark.parametrize("length,width,thick", [(60, 20, 6), (137, 18, 5),
                                                (200, 30, 10)])
def test_link_follows_section(length, width, thick):
    p = parts.link(length, width, thick)
    _ok(p)
    bb = p.bounding_box()
    # end bosses have radius width / 2 around both axes
    d = max(width, parts.END_BOSS_D)
    assert bb.size.X == pytest.approx(length + d, abs=0.2)
    assert bb.size.Y == pytest.approx(d, abs=0.2)
    assert bb.size.Z == pytest.approx(thick, abs=0.1)
    assert bb.min.X == pytest.approx(-d / 2, abs=0.2)


@pytest.mark.parametrize("width", [12, 20, 30])
def test_link_holes_stay_inside_the_material(width):
    import math
    t = 6
    p = parts.link(80, width, t)
    solid = parts._bar(0.0, 80.0, width, t)
    # a hole that breaks out through the rim removes less than its full volume
    full = math.pi * t * (
        parts.HORN_BORE_D ** 2 / 4 + parts.HORN_N_HOLES * parts.HORN_HOLE_D ** 2 / 4
        + parts.HORN_BORE_D ** 2 / 4 + 4 * parts.M3 ** 2 / 4)
    assert solid.volume - p.volume == pytest.approx(full, rel=1e-3)
    for x, y in _mount_points():
        assert _hole_is_free(p, 80 + x, y, t / 2)


def test_link_rejects_short_plate():
    with pytest.raises(ValueError):
        parts.link(10, 20, 6)


@pytest.mark.parametrize("profile,bend", [("L", 30), ("L", -90), ("arc", -45)])
def test_bent_link_follows_its_path(profile, bend):
    p = parts.link(80, 20, 6, profile, bend)
    _ok(p)
    straight = parts.link(80, 20, 6)
    assert p.bounding_box().size.Z == pytest.approx(6, abs=0.1)
    assert p.bounding_box().size.Y > straight.bounding_box().size.Y + 1


@pytest.mark.parametrize("shape", ["sphere", "blade", "pad", "claw"])
def test_foot_shapes_are_solids(shape):
    p = parts.foot(18, shape)
    _ok(p)
    assert p.bounding_box().size.X > 0 and p.bounding_box().size.Y > 0


@pytest.mark.parametrize("distance,angle,width,thick", [(40, 90, 20, 6),
                                                        (60, 45, 18, 5),
                                                        (80, 150, 30, 10)])
def test_elbow_plate_follows_distance(distance, angle, width, thick):
    p = parts.elbow_plate(distance, angle, width, thick)
    _ok(p)
    bb = p.bounding_box()
    d = max(width, parts.END_BOSS_D)
    assert bb.size.Y == pytest.approx(distance + d, abs=0.2)
    assert bb.size.X == pytest.approx(d, abs=0.2)
    assert bb.size.Z == pytest.approx(thick, abs=0.1)
    # the mounting pattern at both ends, the far one turned by angle - 90
    for x, y in _mount_points():
        assert _hole_is_free(p, x, y, thick / 2)
    for x, y in _mount_points(0, distance, angle - 90):
        assert _hole_is_free(p, x, y, thick / 2)


@pytest.mark.parametrize("r", [8, 9, 20, 30])
def test_foot_radius(r):
    p = parts.foot(r)
    _ok(p)
    assert p.bounding_box().size.X >= 2 * min(r, parts.FOOT_MIN_R) - 0.1


@pytest.mark.parametrize("r", [30, 45, 60])
def test_wheel_radius(r):
    p = parts.wheel(r)
    _ok(p)
    assert p.bounding_box().size.X == pytest.approx(2 * r, abs=0.1)


def test_wheel_rejects_bad_radius():
    with pytest.raises(ValueError):
        parts.wheel(20)


@pytest.mark.parametrize("width,blade", [(20, 2), (20, 4), (30, 6), (20, 8)])
def test_spring_flexure(width, blade):
    p = parts.spring(width, blade)
    _ok(p)
    t = max(parts.LINK_T, blade)
    bb = p.bounding_box()
    assert bb.size.X == pytest.approx(parts.SPRING_FLEX + 2 * parts.SPRING_BLOCK,
                                      abs=0.1)
    assert bb.size.Y == pytest.approx(max(width, parts.MOUNT_MIN_W), abs=0.2)
    assert bb.size.Z == pytest.approx(t, abs=0.1)
    # the blade, measured from the part: a slab through the free length
    mid = p & Box(parts.SPRING_FLEX - 2, 200, 200)
    mbb = mid.bounding_box()
    assert mbb.size.Z == pytest.approx(blade, abs=0.05)
    assert mbb.size.Y == pytest.approx(width, abs=0.05)
    assert mid.volume == pytest.approx((parts.SPRING_FLEX - 2) * width * blade, rel=1e-3)
    # the mounting pattern on each block, inside the solid
    xb = (parts.SPRING_FLEX + parts.SPRING_BLOCK) / 2
    for sign in (-1, 1):
        for x, y in _mount_points(sign * xb, 0):
            assert _hole_is_free(p, x, y, t / 2)
            bbp = p.bounding_box()
            assert bbp.min.X < x < bbp.max.X and abs(y) < bbp.size.Y / 2


def test_spring_rejects_a_blade_out_of_range():
    for blade in (1.0, 9.0):
        with pytest.raises(ValueError):
            parts.spring(20, blade)


@pytest.mark.parametrize("length", [40, 100, 200])
def test_tail_bar(length):
    p = parts.tail_bar(length)
    _ok(p)
    bb = p.bounding_box()
    assert bb.size.X == pytest.approx(length, abs=0.1)
    assert bb.size.Y == pytest.approx(parts.TAIL_W, abs=0.1)
    assert bb.size.Z == pytest.approx(parts.TAIL_T, abs=0.1)


def test_tail_bar_rejects_short_or_oversized_bar():
    for length in (10, parts.BED_MM + 1):
        with pytest.raises(ValueError):
            parts.tail_bar(length)

def test_skid_pad():
    p = parts.skid()
    _ok(p)
    bb = p.bounding_box()
    assert bb.size.X == pytest.approx(parts.SKID_L, abs=0.1)
    assert bb.size.Y == pytest.approx(parts.SKID_W, abs=0.1)
    assert bb.size.Z == pytest.approx(parts.SKID_T, abs=0.1)


def test_spike_tip():
    p = parts.spike()
    _ok(p)
    bb = p.bounding_box()
    assert bb.size.X == pytest.approx(parts.SPIKE_L, abs=0.1)
    assert bb.size.Y == pytest.approx(2 * parts.SPIKE_R, abs=0.1)


def test_torso_plate():
    p = parts.torso_plate(_polygon(200, 100), HIPS)
    _ok(p)
    assert p.bounding_box().size.X == pytest.approx(200, abs=0.1)


def test_torso_plate_tabs_carry_edge_brackets():
    # rear and side mounts on the edges: tabs grow the plate by half a bracket,
    # less the axis offset, since the servo body sits behind its axis (inwards)
    p = parts.torso_plate(_polygon(200, 100), [(-100, 0), (70, 50), (70, -50)])
    _ok(p)
    ol = parts.SERVO_L + 2 * parts.SERVO_CLEARANCE + 2 * parts.WALL
    a = parts.SERVO_AXIS_X
    assert p.bounding_box().min.X == pytest.approx(-100 + a - ol / 2, abs=0.1)
    assert p.bounding_box().size.Y == pytest.approx(100 + ol - 2 * a, abs=0.1)


def test_torso_tiles_keeps_a_plate_that_fits():
    tiles = parts.torso_tiles(_polygon(200, 100), HIPS)
    assert len(tiles) == 1
    _ok(tiles[0])
    assert tiles[0].bounding_box().size.X == pytest.approx(200, abs=0.1)


def test_torso_tiles_split_a_large_plate():
    tiles = parts.torso_tiles(_polygon(300, 150), HIPS)
    assert len(tiles) >= 2
    for tile in tiles:
        _ok(tile)
        bb = tile.bounding_box()
        assert bb.size.X <= parts.BED_MM + 0.1
        assert bb.size.Y <= parts.BED_MM + 0.1


def test_splice_plate():
    p = parts.splice_plate(150)
    _ok(p)
    bb = p.bounding_box()
    assert bb.size.X == pytest.approx(parts.SPLICE_W, abs=0.1)
    assert bb.size.Y == pytest.approx(150, abs=0.1)
    assert bb.size.Z == pytest.approx(parts.PLATE_T, abs=0.1)


def test_export_stl(tmp_path):
    f = tmp_path / "x.stl"
    parts.export_stl(parts.foot(), f)
    assert f.stat().st_size > 0


def test_caster_parts():
    _ok(parts.caster_wheel(25))
    _ok(parts.caster_fork())


def test_caster_fork_axle_holes_are_collinear():
    p = parts.caster_fork()
    w = parts.CASTER_GAP + 2 * parts.CASTER_ARM
    arm_x = w / 2 - parts.CASTER_ARM / 2
    y, z = 10.0, parts.FORK_T / 2
    # a pin along X through both arms passes, one shifted 2 mm in Y or Z does not
    assert _hole_is_free(p, 0, y, z, r=1.5, axis="X")
    assert not _hole_is_free(p, 0, y + 2, z, r=1.5, axis="X")
    assert not _hole_is_free(p, 0, y, z + 2, r=1.5, axis="X")
    for sx in (-arm_x, arm_x):
        assert (p & (Pos(sx, y, z) * Box(2, 20, 20))).volume > 0   # arms exist


def test_torso_mount_pattern_follows_the_slot_yaw():
    mounts = [(0, 50, 90 + 30), (60, 0, 0)]
    p = parts.torso_plate(_polygon(200, 100), mounts)
    _ok(p)
    for (x, y, yaw) in mounts:
        for px, py in _mount_points(x, y, yaw):
            assert _hole_is_free(p, px, py, 3.0)
    # the tab footprint is turned too: it reaches further in Y than a 0 deg one
    flat = parts.torso_plate(_polygon(200, 100), [(0, 50, 0)])
    assert p.bounding_box().max.Y > flat.bounding_box().max.Y - 1e-6


def test_star_contour_torso_plate_matches_the_polygon():
    poly = _star()
    p = parts.torso_plate(poly, [])
    _ok(p)
    bb = p.bounding_box()
    assert bb.size.X == pytest.approx(max(x for x, _ in poly) - min(x for x, _ in poly),
                                      abs=0.1)
    assert bb.size.Y == pytest.approx(max(y for _, y in poly) - min(y for _, y in poly),
                                      abs=0.1)

"""Tests for the parametric printable parts."""

import pytest

pytest.importorskip("build123d")

from darwinbot import parts  # noqa: E402

HIPS = [(50, 40), (50, -40), (-50, 40), (-50, -40)]


def _ok(p):
    assert p.is_valid
    assert p.volume > 0


def test_servo_bracket():
    _ok(parts.servo_bracket())


@pytest.mark.parametrize("length,width,thick", [(60, 20, 6), (137, 18, 5),
                                                (200, 30, 10)])
def test_link_follows_section(length, width, thick):
    p = parts.link(length, width, thick)
    _ok(p)
    bb = p.bounding_box()
    # end bosses have radius width / 2 around both axes
    assert bb.size.X == pytest.approx(length + width, abs=0.2)
    assert bb.size.Y == pytest.approx(width, abs=0.2)
    assert bb.size.Z == pytest.approx(thick, abs=0.1)
    assert bb.min.X == pytest.approx(-width / 2, abs=0.2)


def test_link_rejects_short_plate():
    with pytest.raises(ValueError):
        parts.link(10, 20, 6)


@pytest.mark.parametrize("distance,angle,width,thick", [(40, 90, 20, 6),
                                                        (60, 45, 18, 5),
                                                        (80, 150, 30, 10)])
def test_elbow_plate_follows_distance(distance, angle, width, thick):
    p = parts.elbow_plate(distance, angle, width, thick)
    _ok(p)
    bb = p.bounding_box()
    assert bb.size.Y == pytest.approx(distance + width, abs=0.2)
    assert bb.size.X == pytest.approx(width, abs=0.2)
    assert bb.size.Z == pytest.approx(thick, abs=0.1)


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


def test_torso_plate():
    p = parts.torso_plate(200, 100, HIPS)
    _ok(p)
    assert p.bounding_box().size.X == pytest.approx(200, abs=0.1)


def test_torso_plate_tabs_carry_edge_brackets():
    # rear and side mounts on the edges: tabs grow the plate by half a bracket
    p = parts.torso_plate(200, 100, [(-100, 0), (70, 50), (70, -50)])
    _ok(p)
    ol = parts.SERVO_L + 2 * parts.SERVO_CLEARANCE + 2 * parts.WALL
    assert p.bounding_box().min.X == pytest.approx(-100 - ol / 2, abs=0.1)
    assert p.bounding_box().size.Y == pytest.approx(100 + ol, abs=0.1)


def test_torso_tiles_keeps_a_plate_that_fits():
    tiles = parts.torso_tiles(200, 100, HIPS)
    assert len(tiles) == 1
    _ok(tiles[0])
    assert tiles[0].bounding_box().size.X == pytest.approx(200, abs=0.1)


def test_torso_tiles_split_a_large_plate():
    tiles = parts.torso_tiles(300, 150, HIPS)
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

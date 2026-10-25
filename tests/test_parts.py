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


@pytest.mark.parametrize("length", parts.LINK_LENGTHS)
def test_link_axis_distance(length):
    p = parts.link(length)
    _ok(p)
    # end bosses have radius LINK_W / 2 around both axes
    assert p.bounding_box().size.X == pytest.approx(length + parts.LINK_W, abs=0.2)
    assert p.bounding_box().min.X == pytest.approx(-parts.LINK_W / 2, abs=0.2)


def test_elbow_plate_matches_simulation():
    p = parts.elbow_plate()
    _ok(p)
    assert p.bounding_box().size.Y == pytest.approx(parts.ELBOW_MM + parts.LINK_W, abs=0.2)


def test_foot():
    _ok(parts.foot())


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


def test_export_stl(tmp_path):
    f = tmp_path / "x.stl"
    parts.export_stl(parts.foot(), f)
    assert f.stat().st_size > 0

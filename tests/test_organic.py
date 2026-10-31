"""Hollowing a flat part keeps its ends, holes and outline and removes material."""

import pytest

pytest.importorskip("build123d")

from build123d import Vector  # noqa: E402

from darwinbot import organic, parts  # noqa: E402


def test_hollow_link_keeps_ends_and_removes_material():
    link = parts.link(120, 20, 6)
    boss = parts._boss_d(20) / 2 + 1
    hollowed, ratio = organic.hollow(link, (0, 0), (120, 0), boss, cell=1.5)
    assert 0.3 < ratio < 0.9
    assert hollowed.volume == pytest.approx(ratio * link.volume)
    box, orig = hollowed.bounding_box(), link.bounding_box()
    assert (box.min - orig.min).length < 1e-6 and (box.max - orig.max).length < 1e-6
    mid = 3.0                                          # half the thickness
    assert hollowed.is_inside(Vector(0, boss - 2, mid))       # root boss stays solid
    assert hollowed.is_inside(Vector(120, boss - 2, mid))     # tip boss too
    assert not hollowed.is_inside(Vector(0, 0, mid))          # the horn hole stays open

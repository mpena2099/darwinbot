"""The --terrains flag of tools/make_readme_figures.py: one ground per robot."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import make_readme_figures as mrf   # noqa: E402


def test_no_terrains_keeps_the_flat_default():
    assert mrf.parse_terrains(None, ["a", "b"]) is None


def test_one_terrain_per_type_in_order():
    terrains = mrf.parse_terrains("flat,rubble", ["a", "b"])
    assert [t.name for t in terrains] == ["flat", "rubble"]


def test_a_preset_uses_its_first_terrain():
    assert [t.name for t in mrf.parse_terrains("terrain", ["a"])] == ["flat"]


def test_counts_must_match_the_types():
    with pytest.raises(SystemExit, match="one terrain per robot"):
        mrf.parse_terrains("flat,rubble", ["a"])


def test_terrains_need_types():
    with pytest.raises(SystemExit, match="--types"):
        mrf.parse_terrains("flat", None)


def test_unknown_terrain_is_reported():
    with pytest.raises(SystemExit, match="Unknown terrain"):
        mrf.parse_terrains("moon", ["a"])

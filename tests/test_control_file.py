"""Tests for the control file export (no morphology dependency)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from darwinbot.control_file import make_control_file, read_control_file, write_control_file

# (slot, kind, axis) in servo order
LAYOUT = [(2, "joint", "pitch"), (5, "joint", "pitch"), (7, "joint", "roll"),
          (9, "wheel", "wheel"), (11, "wheel", "wheel")]
PARAMS = SimpleNamespace(omega=2 * np.pi * 1.4,
                         amp1=np.array([0.3, 0.3, 0.2, 0, 0]),
                         phase1=np.array([1.0, 2.0, 3.0, 0, 0]),
                         offset=np.array([-0.1, -0.1, 0.3, 4.0, -4.0]),
                         amp2=np.array([0.1, 0.05, 0.0, 0, 0]),
                         phase2=np.array([0.5, 1.5, 2.5, 0, 0]),
                         sigma=np.array([1.0, -1.0, 0.0, 0, 0]))
RANGES = {"pitch": 1.2, "roll": 0.9, "yaw": 0.9}
SIGNS = [1, -1, 1, 1, -1]


def _make():
    return make_control_file("test", 1.5, LAYOUT, PARAMS, RANGES, signs=SIGNS)


def test_round_trip(tmp_path):
    d = _make()
    write_control_file(d, tmp_path / "c.json")
    assert read_control_file(tmp_path / "c.json") == d


def test_ids_names_signs_frequency():
    d = _make()
    s = d["servos"]
    assert d["format"] == 2
    assert [x["id"] for x in s] == [1, 2, 3, 4, 5]
    assert [x["name"] for x in s] == ["s2_pitch", "s5_pitch", "s7_roll", "s9_wheel", "s11_wheel"]
    assert [x["sign"] for x in s] == SIGNS
    assert d["frequency"] == pytest.approx(1.4)
    assert s[0]["min"] == -1.2 and s[0]["max"] == 1.2 and s[0]["zero"] == 2048
    assert s[2]["min"] == -0.9


def test_position_servo_fields():
    s = _make()["servos"][1]
    assert s["mode"] == "position"
    assert (s["amp1"], s["phase1"], s["offset"]) == (0.3, 2.0, -0.1)
    assert (s["amp2"], s["phase2"], s["sigma"]) == (0.05, 1.5, -1.0)
    assert "amp" not in s and "phase" not in s


def test_wheel_fields():
    s = _make()["servos"][4]
    assert s["mode"] == "wheel" and s["speed"] == -4.0
    assert "amp1" not in s and "sigma" not in s


def test_default_signs_are_one():
    d = make_control_file("t", 0, LAYOUT, PARAMS, RANGES)
    assert all(s["sign"] == 1 for s in d["servos"])


@pytest.mark.parametrize("old", [1, 3])
def test_rejects_other_formats(tmp_path, old):
    d = _make()
    d["format"] = old
    with pytest.raises(ValueError, match="format"):
        write_control_file(d, tmp_path / "c.json")
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="only format 2"):
        read_control_file(tmp_path / "bad.json")


def test_rejects_missing_fields(tmp_path):
    for key in ("phase2", "sigma"):
        d = _make()
        del d["servos"][0][key]
        with pytest.raises(ValueError, match=key):
            write_control_file(d, tmp_path / "c.json")
    d = _make()
    del d["frequency"]
    with pytest.raises(ValueError, match="frequency"):
        write_control_file(d, tmp_path / "c.json")

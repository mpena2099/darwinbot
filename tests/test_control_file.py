"""Tests for the control file export (no morphology dependency)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from darwinbot.control_file import make_control_file, read_control_file, write_control_file

LAYOUT = [(0, 0, "hip_pitch"), (0, 1, "hip_pitch"), (0, 0, "knee"), (1, 0, "wheel"), (1, 1, "wheel")]
CPG = SimpleNamespace(omega=2 * np.pi * 1.4,
                      amp=np.array([0.3, 0.3, 0.2, 0, 0]),
                      offset=np.array([-0.1, -0.1, 0.3, 4.0, 4.0]),
                      phase=np.array([1.0, 2.0, 3.0, 0, 0]))
RANGES = {"hip_pitch": 1.2, "knee": 1.3}


def _make():
    return make_control_file("test", 1.5, LAYOUT, CPG, RANGES)


def test_round_trip(tmp_path):
    d = _make()
    write_control_file(d, tmp_path / "c.json")
    assert read_control_file(tmp_path / "c.json") == d


def test_ids_names_signs_frequency():
    d = _make()
    s = d["servos"]
    assert [x["id"] for x in s] == [1, 2, 3, 4, 5]
    assert [x["name"] for x in s] == ["p0L_hip_pitch", "p0R_hip_pitch", "p0L_knee", "p1L_wheel", "p1R_wheel"]
    assert [x["sign"] for x in s] == [1, -1, 1, 1, -1]
    assert d["frequency"] == pytest.approx(1.4)
    assert s[0]["min"] == -1.2 and s[0]["max"] == 1.2 and s[0]["zero"] == 2048
    assert s[3]["mode"] == "wheel" and s[3]["speed"] == 4.0 and "amp" not in s[3]


def test_wheel_sign_right_option():
    d = make_control_file("t", 0, LAYOUT, CPG, RANGES, wheel_sign_right=1)
    assert d["servos"][4]["sign"] == 1


def test_rejects_wrong_format(tmp_path):
    d = _make()
    d["format"] = 2
    with pytest.raises(ValueError, match="format"):
        write_control_file(d, tmp_path / "c.json")
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="format"):
        read_control_file(tmp_path / "bad.json")


def test_rejects_missing_fields(tmp_path):
    d = _make()
    del d["servos"][0]["phase"]
    with pytest.raises(ValueError, match="phase"):
        write_control_file(d, tmp_path / "c.json")
    d = _make()
    del d["frequency"]
    with pytest.raises(ValueError, match="frequency"):
        write_control_file(d, tmp_path / "c.json")

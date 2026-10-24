"""Tests for the robot export folder."""

import csv
import os

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import morphology as mo
from darwinbot.control_file import read_control_file
from darwinbot.evolve import Elite
from darwinbot.export import export_robot


def _elite():
    # three pairs, one wheeled, one with yaw: all branches of the export
    g = np.full(mo.MORPH_LENGTH, 0.5)
    for p in range(mo.N_PAIRS):
        b = 2 + p * mo.PAIR_GENES
        g[b] = 0.9
        g[b + 1] = (-1.0 + 1) / 2 + p * 0.4
        g[b + 5] = 0.9 if p == 0 else 0.1
        g[b + 6] = 0.9 if p == 2 else 0.1
        g[b + 7] = 0.5
    ctrl = np.random.default_rng(0).uniform(0, 1, co.CTRL_LENGTH)
    return Elite(g, ctrl, 1.5, [1.5], 0)


def test_export_files(tmp_path):
    elite = _elite()
    morph = mo.decode(elite.morph)
    out = tmp_path / "robot"
    export_robot(elite, out, "test-robot", stl=False)
    for f in ("control.json", "bom.csv", "bom.md", "assembly.md", "robot.xml"):
        assert (out / f).stat().st_size > 0
    ctl = read_control_file(out / "control.json")
    assert len(ctl["servos"]) == morph.n_servos
    with open(out / "bom.csv") as f:
        rows = list(csv.DictReader(f))
    servo_row = next(r for r in rows if "STS3215" in r["item"])
    assert int(servo_row["quantity"]) == morph.n_servos
    assert any(r["item"].startswith("wheel_") for r in rows)
    assert any(r["item"] == "foot.stl" for r in rows)
    assert "\u2014" not in (out / "assembly.md").read_text()


def test_export_stls(tmp_path):
    pytest.importorskip("build123d")
    out = tmp_path / "robot"
    export_robot(_elite(), out, "test-robot")
    stls = os.listdir(out / "parts")
    assert "torso_plate.stl" in stls and "servo_bracket.stl" in stls
    assert any(s.startswith("link_") for s in stls)
    for s in stls:
        assert (out / "parts" / s).stat().st_size > 0

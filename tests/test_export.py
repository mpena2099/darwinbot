"""Tests for the robot export folder."""

import csv
import os

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import morphology as mo
from darwinbot.control_file import read_control_file
from darwinbot.evolve import Elite
from darwinbot.export import bom_rows, export_robot, mount_positions_mm, parent_chain


def _tree():
    """Two wheeled legs with a yaw joint, one plain leg, plus an explicit foot."""
    m = mo.template(2, wheels=True, yaw=True)
    first = max(m.nodes) + 1
    m.nodes[first] = mo.Node(first, "joint", 0, 0, 0, axis="roll", length=0.10)
    m.nodes[first + 1] = mo.Node(first + 1, "foot", first, 0)
    return m


def _elite(morph=None, seed=0):
    morph = morph or _tree()
    ctrl = np.random.default_rng(seed).uniform(0, 1, co.CTRL_LENGTH)
    return Elite(mo.encode(morph), ctrl, 1.5, [1.5], 0)


def test_export_files(tmp_path):
    elite = _elite()
    morph = mo.decode(elite.morph)
    out = tmp_path / "robot"
    info = export_robot(elite, out, "test-robot", stl=False)
    for f in ("control.json", "bom.csv", "bom.md", "assembly.md", "robot.xml"):
        assert (out / f).stat().st_size > 0
    ctl = read_control_file(out / "control.json")
    assert len(ctl["servos"]) == morph.n_servos == info["servos"]
    assert [s["id"] for s in ctl["servos"]] == list(range(1, morph.n_servos + 1))
    with open(out / "bom.csv") as f:
        rows = list(csv.DictReader(f))
    servo_row = next(r for r in rows if "STS3215" in r["item"])
    assert int(servo_row["quantity"]) == morph.n_servos
    assert any(r["item"].startswith("wheel_") for r in rows)
    assert any(r["item"] == "foot.stl" for r in rows)
    assert not (out / "parts").exists()
    assert "\u2014" not in (out / "assembly.md").read_text()


def test_bom_counts_from_tree():
    morph = _tree()
    rows = {item: q for item, q, _ in bom_rows(morph)}
    links = sum(q for k, q in rows.items() if k.startswith("link_"))
    assert links == morph.n_joints
    wheels = sum(q for k, q in rows.items() if k.startswith("wheel_"))
    assert wheels == morph.n_wheels
    assert rows["servo_bracket.stl"] == morph.n_servos
    # the explicit foot plus the free tip of every joint without a child
    assert rows["foot.stl"] >= 1


def test_assembly_has_one_row_per_servo(tmp_path):
    elite = _elite()
    morph = mo.decode(elite.morph)
    export_robot(elite, tmp_path, "t", stl=False)
    text = (tmp_path / "assembly.md").read_text()
    table = [l for l in text.splitlines() if l.startswith("| ") and l[2].isdigit()]
    assert len(table) == morph.n_servos
    first = morph.layout()[0].slot
    assert parent_chain(morph, first) in table[0]
    assert "core" in table[0]


def test_parent_chain_words():
    m = mo.template(1, wheels=True)
    chain = parent_chain(m, m.layout()[-1].slot)
    assert chain.startswith("core ") and chain.endswith("wheel")
    assert chain.count(">") == 3 and "joint pitch" in chain


def test_mount_positions_follow_used_slots():
    m = mo.template(2)
    pos = mount_positions_mm(m)
    assert len(pos) == len(m.child_map()[0]) == 4
    assert all(abs(abs(y) - m.body_width * 500) < 1e-6 for _, y in pos)


def test_random_trees_export(tmp_path):
    rng = np.random.default_rng(1)
    for k in range(4):
        elite = Elite(mo.random_morphology(rng), co.random_controller(rng), 0.5, [0.5], 0)
        out = tmp_path / str(k)
        info = export_robot(elite, out, f"r{k}", stl=False)
        ctl = read_control_file(out / "control.json")
        assert len(ctl["servos"]) == info["servos"] == mo.decode(elite.morph).n_servos


def test_export_stls(tmp_path):
    pytest.importorskip("build123d")
    out = tmp_path / "robot"
    elite = _elite()
    export_robot(elite, out, "test-robot")
    stls = os.listdir(out / "parts")
    assert "torso_plate.stl" in stls and "servo_bracket.stl" in stls
    assert "foot.stl" in stls
    assert any(s.startswith("link_") for s in stls)
    assert any(s.startswith("wheel_") for s in stls)
    with open(out / "bom.csv") as f:
        listed = {r["item"] for r in csv.DictReader(f) if r["item"].endswith(".stl")}
    assert listed == set(stls)
    for s in stls:
        assert (out / "parts" / s).stat().st_size > 0

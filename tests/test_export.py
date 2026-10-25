"""Tests for the robot export folder."""

import csv
import os

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import morphology as mo
from darwinbot.control_file import read_control_file
from darwinbot.evolve import Elite
from darwinbot.export import (bom_rows, export_robot, mount_positions_mm,
                              parent_chain, part_counts)


def _tree():
    """Wheeled legs, a plain leg, a foot, a spring and one module of every
    new passive terminal kind."""
    m = mo.template(2, wheels=True)
    first = max(m.nodes) + 1
    m.nodes[first] = mo.Node(first, "joint", 0, 6, 0, axis="roll", length=0.10)
    m.nodes[first + 1] = mo.Node(first + 1, "foot", first, 0)
    m.nodes[first + 2] = mo.Node(first + 2, "spring", 0, 1, 0, axis="pitch",
                                 length=0.10, width=0.020, thick=0.004)
    m.nodes[first + 3] = mo.Node(first + 3, "tail", 0, 0, 0, length=0.12,
                                 tail_mass=0.06)
    m.nodes[first + 4] = mo.Node(first + 4, "skid", first, 1)
    m.nodes[first + 5] = mo.Node(first + 5, "spike", 0, 3, 0)
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
    assert any(r["item"].startswith("foot_") for r in rows)
    assert not (out / "parts").exists()
    assert "\u2014" not in (out / "assembly.md").read_text()


def test_bom_counts_from_tree():
    morph = _tree()
    rows = {item: q for item, q, _ in bom_rows(morph)}
    links = sum(q for k, q in rows.items() if k.startswith("link_"))
    assert links == sum(n.kind in ("joint", "link", "spring")
                        for n in morph.nodes.values())
    wheels = sum(q for k, q in rows.items() if k.startswith("wheel_"))
    assert wheels == morph.n_wheels
    assert rows["servo_bracket.stl"] == morph.n_servos
    side = sum(1 for n in morph.nodes.values()
               if n.parent > 0 and n.slot and morph.nodes[n.parent].kind in ("joint", "link"))
    assert side > 0
    assert sum(q for k, q in rows.items() if k.startswith("elbow_")) == side
    # the explicit foot plus the free tip of every joint without a child
    assert sum(q for k, q in rows.items() if k.startswith("foot_")) >= 1


def test_bom_links_match_distinct_sizes():
    morph = _tree()
    joint = next(n for n in morph.nodes.values() if n.kind == "joint")
    joint.width, joint.thick = 0.017, 0.005
    rows = {item: q for item, q, _ in bom_rows(morph)}
    expected = {f"link_{int(round(n.length * 1000))}x{int(round(n.width * 1000))}"
                f"x{int(round(mo.plate_thick(n) * 1000))}.stl"
                for n in morph.nodes.values()
                if n.kind in ("joint", "link", "spring")}
    assert len(expected) > 1
    assert {k for k in rows if k.startswith("link_")} == expected
    assert all(rows[name] >= 1 for name in expected)


def test_bom_counts_springs_as_tpu_and_pla():
    morph = _tree()
    rows = {item: (q, note) for item, q, note in bom_rows(morph)}
    assert rows["spring_20x4.stl"] == (1, "TPU, flexure")
    assert rows["link_100x20x6.stl"][0] >= 1      # its PLA segment, 6 mm thick


def test_bom_counts_terminal_modules():
    morph = _tree()
    rows = {item: (q, note) for item, q, note in bom_rows(morph)}
    assert rows["tail_120.stl"] == (1, "PLA")
    assert rows["skid.stl"] == (1, "PLA, low-friction pad")
    assert rows["spike.stl"] == (1, "TPU, high-friction tip")
    qty, note = rows["M3 nuts as counterweight"]
    assert qty == 1 and "60.0 g" in note


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
    assert "servo_bracket.stl" in stls
    assert any(s.startswith("torso_") for s in stls)
    assert any(s.startswith("foot_") for s in stls)
    assert any(s.startswith("link_") for s in stls)
    assert any(s.startswith("elbow_") for s in stls)
    assert any(s.startswith("spring_") for s in stls)
    assert any(s.startswith("wheel_") for s in stls)
    assert "tail_120.stl" in stls and "skid.stl" in stls and "spike.stl" in stls
    with open(out / "bom.csv") as f:
        listed = {r["item"] for r in csv.DictReader(f) if r["item"].endswith(".stl")}
    assert listed == set(stls)
    for s in stls:
        assert (out / "parts" / s).stat().st_size > 0


def test_large_torso_splits_into_tiles_with_splice_plates():
    m = mo.template(3, wheels=True)
    m.body_length, m.body_width = 0.36, 0.22
    rows = {item: q for item, q, _ in bom_rows(m)}
    tiles = [k for k in rows if k.startswith("torso_tile_")]
    splices = [k for k in rows if k.startswith("splice_plate_")]
    assert len(tiles) >= 2 and splices
    # splice length in mm, not metres scaled twice
    assert all(int(k.split("_")[2].split(".")[0]) < 250 for k in splices)

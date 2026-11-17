"""
test_robot_file.py
------------------
Robots written by hand in YAML: round trip, validation and locked parts.
"""

import os
import pickle

import numpy as np
import pytest
import yaml

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo
from darwinbot import robot_file as rf

CAR = os.path.join(os.path.dirname(__file__), "..", "examples", "robots",
                   "two-wheel-car.yaml")


def _doc(**node):
    return {"torso": {"length_mm": 160, "width_mm": 120},
            "nodes": {"n": {"parent": "torso", "slot": 2, **node}}}


def test_dump_load_round_trip_keeps_the_genome():
    for seed in range(8):
        genes = mo.encode(mo.decode(mo.random_morphology(np.random.default_rng(seed))))
        morph = mo.decode(genes)
        back, locked = rf.parse(yaml.safe_load(rf.dump(morph, (0,))))
        assert locked == (0,)
        assert np.allclose(mo.encode(back), genes, atol=1e-6)


def test_car_example_loads_with_locked_wheels():
    morph, locked = rf.load(CAR)
    assert locked == (0, 1, 2)
    assert [morph.nodes[i].kind for i in (1, 2)] == ["wheel", "wheel"]
    assert morph.nodes[3].kind == "caster"
    assert morph.nodes[1].wheel_rad == pytest.approx(0.06)
    assert morph.body_length == pytest.approx(0.16)


@pytest.mark.parametrize("node, word", [
    (dict(kind="wheel", radius_mm=150), "radius_mm"),
    (dict(kind="joint", length_mm=10), "length_mm"),
    (dict(kind="joint", axis="spin"), "axis"),
    (dict(kind="wheel", colour="red"), "colour"),
    (dict(kind="wheel", length_mm=80), "length_mm"),
    (dict(kind="engine"), "kind"),
    (dict(kind="joint", slot=40), "slot"),
    (dict(kind="foot", shape="hoof"), "shape"),
    (dict(kind="link", profile="L", bend_deg=120), "bend_deg"),
])
def test_bad_values_name_the_field(node, word):
    node = {"kind": "joint", **node}
    with pytest.raises(ValueError, match=word):
        rf.parse(_doc(**node))


def test_bad_structure_is_refused():
    with pytest.raises(ValueError, match="parent"):
        rf.parse(_doc(kind="foot", parent="ghost"))
    with pytest.raises(ValueError, match="torso length_mm"):
        rf.parse({"torso": {"length_mm": 900}})
    with pytest.raises(ValueError, match="unknown key"):
        rf.parse({"torso": {}, "nodes": {}, "extra": 1})
    doc = _doc(kind="wheel")
    doc["nodes"]["m"] = {"parent": "torso", "slot": 2, "kind": "wheel"}
    with pytest.raises(ValueError, match="already used"):
        rf.parse(doc)
    doc = {"torso": {}, "nodes": {f"n{i}": {"parent": "torso", "slot": 0, "kind": "link"}
                                  for i in range(mo.MAX_NODES)}}
    with pytest.raises(ValueError, match="MAX_NODES"):
        rf.parse(doc)
    doc = {"torso": {}, "nodes": {"w": {"parent": "torso", "slot": 2, "kind": "wheel"},
                                  "f": {"parent": "w", "slot": 0, "kind": "foot"}}}
    with pytest.raises(ValueError, match="no slots"):
        rf.parse(doc)


def test_mutation_filtered_as_evolve_keeps_locked_parts():
    morph, locked = rf.load(CAR)
    parent = mo.encode(morph)
    rng = np.random.default_rng(3)
    kept, changed = 0, 0
    for _ in range(300):
        child = mo.mutate(parent, rng)
        if not mo.keeps_locked(child, parent, locked):
            continue
        kept += 1
        a, b = mo.decode(child), mo.decode(parent)
        for i in (1, 2):
            assert a.nodes[i] == b.nodes[i]
        assert (a.body_length, a.body_width, a.contour) == (
            b.body_length, b.body_width, b.contour)
        changed += not np.allclose(child, parent)
    assert kept > 0 and changed > 0
    # something was rejected too: the filter does work
    assert kept < 300


def test_run_on_a_seeded_archive_keeps_locked_wheels(tmp_path):
    morph, locked = rf.load(CAR)
    archive = evolve.Archive(terrains="flat", seconds=2.0, variants=0, task="forward",
                             locked=locked)
    genes = mo.encode(morph)
    morph = mo.decode(genes)          # the body as the genome holds it
    archive.insert(evolve.Elite(genes, np.zeros(co.CTRL_LENGTH), 1.0, [1.0], 0,
                                np.array([0.5, 0.05, 2.0, 1.0, 1.0, 0.0])))
    path = tmp_path / "seed" / "archive.pkl"
    path.parent.mkdir()
    with open(path, "wb") as f:
        pickle.dump(archive, f)
    out = evolve.run(n_batches=2, batch_size=4, budget_child=4, workers=1,
                     resume=str(path), out_dir=str(tmp_path / "out"))
    assert len(out.history) >= 1 and out.locked == locked
    for e in out.cells.values():
        m = mo.decode(e.morph)
        for i in (1, 2):
            assert m.nodes[i] == morph.nodes[i]
        assert m.body_length == pytest.approx(morph.body_length)


def test_replay_robot_stands_still():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import replay
    from darwinbot.sim import Episode
    from darwinbot.terrain import get_terrains
    morph, _ = rf.load(os.path.join(os.path.dirname(__file__), "..", "examples",
                                    "robots", "two-wheel-car.yaml"))
    ep = Episode(morph, get_terrains("flat")[0], 2.0)
    m = ep.run(replay.standing(morph))
    assert ep.buildable and not m.fell
    assert abs(m.dx) < 0.01 and abs(m.yaw) < 0.02

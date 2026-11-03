"""
test_check_assembly.py
----------------------
The virtual assembly of a robot: printed parts and servo bodies must not
overlap, and link holes must line up with the servo horn holes they bolt to.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("build123d")
pytest.importorskip("mujoco")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import assembly                                     # noqa: E402
import check_assembly as ca                         # noqa: E402

from darwinbot import controller as co              # noqa: E402
from darwinbot import evolve, morphology as mo      # noqa: E402


def _archive(tmp_path):
    """A one cell archive with a two pair template body, and its type name."""
    morph = mo.template(2)
    ctrl = np.random.default_rng(0).uniform(0, 1, co.CTRL_LENGTH)
    archive = evolve.Archive(terrains="flat")
    archive.insert(evolve.Elite(mo.encode(morph), ctrl, 1.0, [1.0], 0))
    path = tmp_path / "archive.pkl"
    with open(path, "wb") as f:
        pickle.dump(archive, f)
    return path, archive.names()[next(iter(archive.cells))]


def test_clean_body_passes(tmp_path, capsys):
    path, name = _archive(tmp_path)
    assert ca.main(["--archive", str(path), "--type", name]) == 0
    out = capsys.readouterr().out
    assert "0 clash(es)" in out and "0 flagged" in out
    assert name in out


def _two_children_on_one_link():
    """The template with a second side module on the first thigh: slots 1 and 2
    of one link both carry a servo."""
    morph = mo.template(2)
    i = max(morph.nodes) + 1
    morph.nodes[i] = mo.Node(i, "joint", 1, 2, 0, axis="pitch", length=0.10)
    return morph


@pytest.mark.parametrize("make", [lambda: mo.template(2),
                                  _two_children_on_one_link,
                                  lambda: mo.template(2, yaw=True),
                                  lambda: mo.template(2, wheels=True)])
def test_clean_templates_clear_at_spawn_and_while_walking(make):
    morph = make()
    ctrl = np.random.default_rng(0).uniform(0, 1, co.CTRL_LENGTH)
    frames = [ca.spawn_frames(morph)] + ca.walk_frames(morph, ctrl)
    assert len(frames) == 1 + len(ca.WALK_TIMES)
    for pose in frames:
        clashes, _ = ca.report(morph, frames=pose)
        assert clashes == []


def test_a_walk_report_covers_the_whole_walk():
    morph = mo.template(2)
    ctrl = np.random.default_rng(0).uniform(0, 1, co.CTRL_LENGTH)
    clashes, horns = ca.report(morph, ctrl=ctrl, walk=True)
    assert clashes == []
    assert len(horns) == morph.n_servos


def test_every_pair_is_checked_including_parent_and_child():
    # a link pushed into the servo it turns about is a clash, not a joint
    morph = mo.template(2)
    records = assembly.placements(morph)
    k = next(i for i, p in enumerate(records) if p.label == "link1")
    records[k] = records[k]._replace(pos=(-20.0, 0.0, 0.0))
    clashes, _ = ca.report(morph, records=records)
    assert any("link1" in (a, b) for a, b, _ in clashes)


def test_a_lap_is_allowed_only_between_screwed_parts():
    morph = mo.template(2)
    pairs = ca.lap_pairs(morph, assembly.placements(morph))
    assert frozenset(("cradle1", "servo1")) in pairs
    assert frozenset(("link1", "servo1")) not in pairs
    assert frozenset(("link1", "link2")) not in pairs


def test_horn_patterns_line_up_on_a_clean_body():
    morph = mo.template(2)
    clashes, horns = ca.report(morph)
    assert clashes == []
    assert len(horns) == morph.n_servos            # one link per servo
    assert max(gap for _, gap in horns) < ca.MAX_GAP


def test_forced_overlap_is_flagged():
    morph = mo.template(2)
    records = assembly.placements(morph)
    k = next(i for i, p in enumerate(records) if p.label.startswith("foot"))
    # push that foot down into the middle of the torso plate
    records[k] = records[k]._replace(body="torso", pos=(0.0, 0.0, -24.0),
                                     rot=np.eye(3))
    clashes, _ = ca.report(morph, records=records)
    assert any({records[k].label, "torso_plate"} == {a, b} for a, b, _ in clashes)


def test_rotated_horn_pattern_is_flagged():
    morph = mo.template(2)
    records = assembly.placements(morph)
    k = next(i for i, p in enumerate(records) if p.horn)
    a = np.radians(30.0)
    turn = np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    records[k] = records[k]._replace(rot=records[k].rot @ turn)
    _, horns = ca.report(morph, records=records)
    assert dict(horns)[records[k].label] > ca.MAX_GAP

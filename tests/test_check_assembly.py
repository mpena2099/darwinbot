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


def _evolved_body(seed):
    """A body made the way evolution makes them: a random seed and mutations."""
    rng = np.random.default_rng(seed)
    genes = mo.random_morphology(rng)
    for _ in range(int(rng.integers(0, 25))):
        genes = mo.mutate(genes, rng)
    return mo.decode(genes)


def test_bodies_the_generator_makes_are_accepted_and_assemble():
    """Most of what random_morphology and mutate make is buildable, and of the
    bodies without a spring (whose printed plates are not drawn to stack yet)
    most have no printed clash at the spawn pose."""
    from darwinbot.sim import Episode
    from darwinbot.terrain import get_terrains
    flat = get_terrains("flat")[0]
    accepted, clean, checked = 0, 0, 0
    for seed in range(14):
        body = _evolved_body(seed)
        if not Episode(body, flat, seconds=0.1).buildable:
            continue
        accepted += 1
        if any(n.kind == "spring" for n in body.nodes.values()):
            continue
        checked += 1
        clashes, _ = ca.report(body)
        clean += not clashes
    assert accepted >= 10
    assert checked and clean >= 0.6 * checked


def _random_bodies(seeds=range(12)):
    out = []
    for s in seeds:
        morph = mo.decode(mo.random_morphology(np.random.default_rng(s)))
        if any(n.kind in ("foot", "link") for n in morph.nodes.values()):
            out.append(pytest.param(morph, id=f"seed{s}"))
    return out


def _sim_world(morph):
    import mujoco
    from darwinbot import builder
    xml, _ = builder.build_mjcf(morph)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("morph", _random_bodies())
def test_foot_placements_sit_at_their_simulated_foot(morph):
    model, data = _sim_world(morph)
    feet = [p for p in assembly.placements(morph) if p.label.startswith("foot")]
    for p in feet:
        b = model.body(p.body).id
        world = data.xpos[b] + data.xmat[b].reshape(3, 3) @ (np.array(p.pos) / 1000)
        # the sim geom the printed foot stands for: the cap sphere of a free link
        # end, or the foot module's own geoms (the body origin)
        node = morph.nodes[p.owner]
        if node.kind == "foot":
            expect = data.xpos[b]
        else:
            expect = data.geom_xpos[model.geom(f"cap{p.owner}").id]
        assert np.linalg.norm(world - expect) < 0.002, p.label


@pytest.mark.parametrize("morph", _random_bodies())
def test_a_printed_link_holds_the_links_rigid_with_it(morph):
    """A plain link hung on another prints as one piece with it: the solid must
    reach the body origin of each such link and of the foot at its end."""
    model, data = _sim_world(morph)
    for p in assembly.placements(morph):
        if not p.label.startswith("link"):
            continue
        owner = p.owner
        b = model.body(p.body).id
        inv = data.xmat[b].reshape(3, 3).T
        solid = assembly.solid(morph, p)
        box = solid.bounding_box()
        for i, n in morph.nodes.items():
            if i == owner or export_owner(morph, i) != owner:
                continue
            local = inv @ (data.xpos[model.body(f"n{i}").id] - data.xpos[b]) * 1000
            near = [s for s in solid.solids()
                    if s.bounding_box().min.X - 1 <= local[0] <= s.bounding_box().max.X + 1]
            assert near and box.min.X - 1 <= local[0] <= box.max.X + 1, p.label


def export_owner(morph, i):
    from darwinbot import export
    return export.print_owner(morph, i)


def test_rolled_rigid_link_is_fused_not_dropped():
    from darwinbot import parts
    rot = (1.0, 0.0, 0.0, 0.0, -0.7795, 0.6264, 0.0, -0.6264, -0.7795)
    child = (105, 20, 6, "straight", 0, None, None, None, (None, None, (), ()))
    solid = parts.link(95, 20, 6, "straight", 0, axis="pitch", foot_r=None,
                       tip_axis=None,
                       elbow=(None, None, (), (((94.7, 0.0, 0.0), rot, child),)))
    assert solid.bounding_box().max.X > 200
    assert len(solid.solids()) == 1

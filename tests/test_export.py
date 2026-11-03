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
                              parent_chain, part_counts, servo_signs)


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
    # a standalone cradle for every servo on the torso; the others are printed
    # with the link tip or the elbow plate that carries them
    on_torso = sum(1 for n in morph.nodes.values()
                   if n.parent in (0,) and n.kind in ("joint", "wheel", "segment"))
    assert sum(q for k, q in rows.items() if k.startswith("cradle_")) == on_torso
    assert rows["M2 x 6 screw"] == 8 * morph.n_servos
    # elbow plates carry the servos on the side of a link; passive side modules
    # are printed with the link
    side = sum(1 for n in morph.nodes.values()
               if n.parent > 0 and n.slot and n.kind in ("joint", "wheel", "segment")
               and morph.nodes[n.parent].kind in ("joint", "link"))
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
                f"x{int(round(mo.plate_thick(n) * 1000))}"
                for n in morph.nodes.values()
                if n.kind in ("joint", "link", "spring")}
    assert len(expected) > 1
    import re
    # a suffix marks the fork root, the tip cradle, the foot and the elbow trims
    sizes = {re.match(r"link_\d+x\d+x\d+", k).group(0)
             for k in rows if k.startswith("link_")}
    assert sizes == expected
    assert all(any(k.startswith(name) for k in rows) for name in expected)


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
    assert all(abs(abs(y) - m.body_width * 500) < 1e-6 for _, y, _ in pos)
    # every mount faces outward, along the radius of its contour point
    for x, y, yaw in pos:
        assert yaw == pytest.approx(np.degrees(np.arctan2(y, x)))


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
    assert any(s.startswith("cradle_") for s in stls)
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


def test_bom_lists_caster_parts():
    m = mo.template(2)
    i = max(m.nodes) + 1
    m.nodes[i] = mo.Node(i, "caster", 0, 1, 0.0, wheel_rad=0.025)
    rows = {item: q for item, q, _ in bom_rows(m)}
    assert rows["caster_wheel_25.stl"] == 1 and rows["caster_fork.stl"] == 1


def test_spring_stl_and_bom_use_the_same_key():
    m = _tree()
    first = max(m.nodes) + 1
    m.nodes[first] = mo.Node(first, "spring", 0, 4, 0, axis="pitch",
                             length=0.14, width=0.020, thick=0.004)
    rows = {item: q for item, q, _ in bom_rows(m)}
    assert rows["spring_20x4.stl"] == 2          # two lengths, one printed part
    assert not any(k.startswith("spring_") and k != "spring_20x4.stl" for k in rows)


def test_bom_lists_screw_lengths_nuts_and_the_placeholder_board():
    rows = {item: (q, note) for item, q, note in bom_rows(_tree())}
    screws = {k: q for k, (q, _) in rows.items() if k.startswith("M3 x ") and k.endswith("screw")}
    assert screws and all(k.split()[2].isdigit() for k in screws)
    assert rows["M3 nut"][0] == sum(screws.values())
    assert "PLACEHOLDER" in rows["ESP32 servo driver board"][1]


@pytest.mark.parametrize("slot", [0, 1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("rolled", [0.0, np.pi])
def test_wheel_sign_follows_the_simulation_axle(slot, rolled):
    """Positive servo turn is about the hub direction; the simulation turns the
    wheel about its local axle. The sign is +1 exactly when they agree."""
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "wheel", 0, slot, rolled, wheel_rad=0.05)
    layout = m.layout()
    sign = servo_signs(m, layout)[0]
    pos, rot = m.frames()[1]
    hub = m.wheel_hub(m.nodes[1], pos, rot)
    world_axle = rot @ np.array([0.0, m.wheel_axis_sign(rot), 0.0])
    world_hub = rot @ np.array([0.0, np.sign(hub[1]), 0.0])
    assert sign == int(round(np.dot(world_axle, world_hub)))
    assert sign in (-1, 1)


def test_rolled_wheel_keeps_its_sign():
    """Rolling the slot 180 degrees flips the axle and the hub side together."""
    for slot in (0, 2, 5):
        signs = []
        for rolled in (0.0, np.pi):
            m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
            m.nodes[1] = mo.Node(1, "wheel", 0, slot, rolled, wheel_rad=0.05)
            signs.append(servo_signs(m, m.layout())[0])
        assert signs[0] == signs[1]


def test_load_full_scale_is_written(tmp_path):
    export_robot(_elite(), tmp_path, "t", stl=False)
    assert read_control_file(tmp_path / "control.json")["load_full_scale"] == 0.7


def _profiled_tree():
    """An L link, an arc link and one explicit foot of every shape."""
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    i = 1
    for slot, profile, shape in ((0, "L", "sphere"), (1, "arc", "blade")):
        m.nodes[i] = mo.Node(i, "joint", 0, slot, 0.0, "pitch", 0.06)
        m.nodes[i + 1] = mo.Node(i + 1, "link", i, 0, 0.0, length=0.08,
                                 profile=profile, bend=np.pi / 2)
        m.nodes[i + 2] = mo.Node(i + 2, "foot", i + 1, 0, 0.0,
                                 foot_rad=0.018, foot_shape=shape)
        i += 3
    m.nodes[i] = mo.Node(i, "foot", 0, 2, 0.0, foot_rad=0.018, foot_shape="pad")
    m.nodes[i + 1] = mo.Node(i + 1, "foot", 0, 3, 0.0, foot_rad=0.018,
                             foot_shape="claw")
    return m


def test_bent_links_and_foot_shapes_export_non_empty_stls(tmp_path):
    pytest.importorskip("build123d")
    out = tmp_path / "robot"
    export_robot(_elite(_profiled_tree()), out, "profiled")
    stls = set(os.listdir(out / "parts"))
    # the plain link hung on each joint is printed as one piece with it, foot
    # spigot included
    merged = [n for n in stls if n.startswith("link_60x20x6_u") and "_k" in n]
    assert len(merged) == 2 and not any(n.startswith("link_80x20x6") for n in stls)
    assert {"foot_18.stl", "foot_blade_18.stl", "foot_pad_18.stl",
            "foot_claw_18.stl"} <= stls
    for name in stls:
        assert (out / "parts" / name).stat().st_size > 0


def test_segment_plate_and_spine_servo_row(tmp_path):
    m = mo.template(2)
    first = max(m.nodes) + 1
    m.nodes[first] = mo.Node(first, "segment", 0, 1, 0, axis="pitch",
                             length=0.12, width=0.10)
    m.nodes[first + 1] = mo.Node(first + 1, "joint", first, 3, 0, axis="pitch",
                                 length=0.08)
    elite = _elite(m)
    morph = mo.decode(elite.morph)
    seg = next(i for i, n in morph.nodes.items() if n.kind == "segment")
    rows = {item: q for item, q, _ in bom_rows(morph)}
    assert rows["segment_plate_120x100.stl"] == 1
    assert rows["Feetech STS3215 servo, 12 V (C018)"] == morph.n_servos
    assert "spine servo" in parent_chain(morph, seg)
    out = tmp_path / "robot"
    export_robot(elite, out, "spine", stl=True)
    assert (out / "parts" / "segment_plate_120x100.stl").stat().st_size > 0
    text = (out / "assembly.md").read_text()
    assert text.count("spine servo") >= 1
    ctl = read_control_file(out / "control.json")
    assert len(ctl["servos"]) == morph.n_servos


def test_face_mounts_export(tmp_path):
    m = mo.Morphology(0.30, 0.20, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "joint", 0, 8, 0.0, angle=0.7)
    m.nodes[2] = mo.Node(2, "joint", 0, 11, 0.0)
    m.nodes[3] = mo.Node(3, "joint", 0, 3, 3 * np.pi / 2)
    pos = mount_positions_mm(m)
    assert [len(p) for p in pos] == [3, 4, 4]       # side slot 3, then 8 and 11
    assert "top front" in parent_chain(m, 1) and "bottom rear" in parent_chain(m, 2)
    pytest.importorskip("build123d")
    out = tmp_path / "robot"
    export_robot(_elite(m), out, "face-robot")
    assert any(s.startswith("torso_") for s in os.listdir(out / "parts"))
    text = (out / "assembly.md").read_text()
    assert "top front" in text and "bottom rear" in text


def _rolled_hips(roll_deg):
    m = mo.template(2)
    for i, n in m.nodes.items():
        if n.parent == 0:
            n.rot = np.radians(roll_deg)
    return m


def test_wedge_only_for_an_edge_roll_off_a_right_angle():
    from darwinbot.export import torso_mount
    plain = _rolled_hips(90.0)
    seats = [torso_mount(plain, i) for i, n in plain.nodes.items() if n.parent == 0]
    assert all(m.kind == "side" and m.angle == 0.0 for m in seats)
    rolled = _rolled_hips(70.0)
    seats = [torso_mount(rolled, i) for i, n in rolled.nodes.items()
             if n.parent == 0]
    assert all(m.kind == "side" and 15.0 < abs(m.angle) < 25.0 and m.drop > 0
               and m.shift for m in seats)
    rows = {item: q for item, q, _ in bom_rows(rolled)}
    assert sum(q for k, q in rows.items() if k.startswith("wedge_")) == 4
    # a face mount never gets one
    m = mo.Morphology(0.30, 0.20, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "joint", 0, 8, 0.4, angle=0.7)
    assert torso_mount(m, 1).kind == "free"
    assert not any(r[0].startswith("wedge_") for r in bom_rows(m))
    # nor does a segment plate: a servo on it is a plain mount
    seg = mo.template(2)
    first = max(seg.nodes) + 1
    seg.nodes[first] = mo.Node(first, "segment", 0, 1, 0, axis="pitch",
                               length=0.12, width=0.10)
    seg.nodes[first + 1] = mo.Node(first + 1, "joint", first, 3, 3 * np.pi / 2,
                                   axis="pitch", length=0.08)
    assert torso_mount(seg, first + 1).angle == 0.0
    assert not any(r[0].startswith("wedge_") for r in bom_rows(seg))


def test_vertical_axis_hips_bolt_down_behind_the_servo(tmp_path):
    from darwinbot.export import torso_mount
    yaw = mo.template(2, yaw=True)
    hips = [i for i, n in yaw.nodes.items() if n.parent == 0]
    assert hips and all(torso_mount(yaw, i).kind == "back" for i in hips)
    assert all(len(p) == 5 for p in mount_positions_mm(yaw))
    pytest.importorskip("build123d")
    export_robot(_elite(yaw), tmp_path / "r", "yaw")
    assert "cradle_yaw_back.stl" in os.listdir(tmp_path / "r" / "parts")


def test_rolled_mount_exports_a_wedge_and_a_shorter_cradle(tmp_path):
    pytest.importorskip("build123d")
    out = tmp_path / "robot"
    export_robot(_elite(_rolled_hips(70.0)), out, "rolled")
    stls = set(os.listdir(out / "parts"))
    assert any(s.startswith("wedge_") for s in stls)
    assert any(s.startswith("cradle_pitch_w") for s in stls)
    assert "wedge" in (out / "assembly.md").read_text()


def _link_on_joint(roll=0.0):
    """A hip joint with a plain link on its tip and a foot on that."""
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, "joint", 0, 3, 3 * np.pi / 2, "pitch", 0.09)
    m.nodes[2] = mo.Node(2, "link", 1, 0, roll, length=0.08)
    m.nodes[3] = mo.Node(3, "foot", 2, 0, 0.0, foot_rad=0.012)
    return m


@pytest.mark.parametrize("roll", [0.0, 1.0])
def test_a_plain_link_on_a_joint_is_printed_with_it(roll, tmp_path):
    from darwinbot.export import print_owner
    m = _link_on_joint(roll)
    assert print_owner(m, 2) == 1 and print_owner(m, 1) == 1
    rows = {item: q for item, q, _ in bom_rows(m)}
    links = [k for k in rows if k.startswith("link_")]
    assert len(links) == 1 and "_k" in links[0]
    pytest.importorskip("build123d")
    export_robot(_elite(m), tmp_path / "r", "merged")
    parts_ = os.listdir(tmp_path / "r" / "parts")
    assert links[0] in parts_ and not any(p.startswith("link_80") for p in parts_)


def test_elbow_trim_follows_distance_angle_and_roll():
    from darwinbot.export import elbow_trim
    at90, x0, x1 = elbow_trim(60, 90, "pitch")
    assert (x0, x1) == (-17, 17) or (x0 <= -16 and x1 >= 16)
    at40 = elbow_trim(40, 90, "pitch")[0]
    assert at90 - at40 == 20
    # a slanted child reaches back further and along the link
    at30, y0, y1 = elbow_trim(60, 30, "pitch")
    assert at30 != at90 and y1 - y0 > x1 - x0
    # never under the thinnest fin, whatever the cradle does
    assert elbow_trim(25, 30, "pitch")[0] == 1


def test_mount_rule_matches_the_exported_cradles():
    # morphology.mount_tilt (the simulation's rule) and export.torso_mount
    # (the printed foot or wedge) must agree on which mounts cannot be bolted
    from darwinbot import export
    for seed in range(40):
        rng = np.random.default_rng(seed)
        g = mo.random_morphology(rng)
        for _ in range(int(rng.integers(0, 15))):
            g = mo.mutate(g, rng)
        m = mo.decode(g)
        for i, node in m.nodes.items():
            if not i or (node.parent == 0 and node.slot in mo.FACE_SLOTS):
                continue
            t, mount = mo.mount_tilt(m, i), export.torso_mount(m, i)
            assert (t is None) == (mount is None)
            if t is None:
                continue
            rule = t > mo.MAX_MOUNT_TILT or (
                t >= 0.5 and node.kind != "wheel" and node.axis == "roll")
            assert rule == (mount.kind == "free")

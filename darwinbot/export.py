"""
export.py
---------
Evolved robot -> build folder for the real DarwinBot: control.json, STL parts, bill of materials, assembly
notes and the MJCF used in simulation.

The body is a module tree (darwinbot/morphology.py), so everything is read by
walking the tree: one STL per distinct part (torso plate with a mount at each
used core slot, a link per catalogue length, servo bracket, wheel per radius,
foot), and one assembly row per servo in layout order. Mount positions on the
plate are the core slot offsets of Morphology.slot_frame, in mm.
"""

import csv
import os
from collections import Counter

from darwinbot import builder
from darwinbot import morphology as morpho
from darwinbot.control_file import make_control_file, write_control_file
from darwinbot.controller import Controller
from darwinbot.terrain import get_terrains

CORE_SLOTS = ("front", "back", "left-front", "left-middle", "left-rear",
              "right-front", "right-middle", "right-rear")
ELBOW_SLOTS = {1: "left elbow", 2: "right elbow"}


def _mm(m):
    return int(round(m * 1000))


def mount_positions_mm(morph):
    """(x, y) on the torso plate of every used core slot, in mm."""
    out = []
    for slot in sorted(morph.child_map().get(0, {})):
        off, _ = morph.slot_frame(morph.nodes[0], slot)
        out.append((float(off[0]) * 1000, float(off[1]) * 1000))
    return out


def part_counts(morph):
    """Printed parts needed: link lengths (mm), wheel radii (mm), feet and
    elbow plates (one per module on a side slot of a link)."""
    cm = morph.child_map()
    links, wheels, feet, elbows = Counter(), Counter(), 0, 0
    for i, n in morph.nodes.items():
        if n.kind in ("joint", "link"):
            links[_mm(n.length)] += 1
            elbows += sum(1 for slot in cm.get(i, {}) if slot)
            if i not in cm:
                feet += 1            # a free tip ends in an implicit foot
        elif n.kind == "wheel":
            wheels[_mm(n.wheel_rad)] += 1
        elif n.kind == "foot":
            feet += 1
    return links, wheels, feet, elbows


def bom_rows(morph):
    """Bill of materials as (item, quantity, note) rows."""
    n = morph.n_servos
    links, wheels, n_feet, n_elbows = part_counts(morph)
    rows = [("Feetech STS3215 servo, 7.4 V", n, "bus servo, ID 1..%d" % n),
            ("torso_plate.stl", 1, "PLA")]
    for length, q in sorted(links.items()):
        rows.append((f"link_{length}.stl", q, "PLA"))
    rows.append(("servo_bracket.stl", n, "PLA, one per servo"))
    if n_elbows:
        rows.append(("elbow_plate.stl", n_elbows, "PLA, one per side module"))
    if n_feet:
        rows.append(("foot.stl", n_feet, "TPU"))
    for r, q in sorted(wheels.items()):
        rows.append((f"wheel_{r}.stl", q, "PLA"))
        rows.append((f"Rubber O-ring tyre, wheel_{r}", q, f"fits rim of radius {r} mm"))
    rows += [("ESP32 servo driver board", 1, "see docs/hardware-dados.md"),
             ("2S LiPo battery 7.4 V", 1, "1000 to 1500 mAh"),
             ("Step-down regulator", 1, "from 7.4 V to the board supply"),
             ("M3 screws", 4 * n + 4 * n_elbows + 4,
              "estimate: 4 per servo, 4 per elbow plate, 4 for the board")]
    return rows


def printed_parts(rows):
    """Number of printed pieces (rows that are .stl files)."""
    return sum(q for item, q, _ in rows if item.endswith(".stl"))


def _write_bom(rows, out_dir, title):
    with open(os.path.join(out_dir, "bom.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "quantity", "note"])
        w.writerows(rows)
    lines = [f"# Bill of materials: {title}", "",
             "| Item | Qty | Note |", "|---|---|---|"]
    lines += [f"| {i} | {q} | {n} |" for i, q, n in rows]
    with open(os.path.join(out_dir, "bom.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def parent_chain(morph, node_id):
    """Path from the core to a node in words, e.g. 'core right-front > joint pitch'."""
    path = []
    i = node_id
    while i != 0:
        path.append(i)
        i = morph.nodes[i].parent
    words = []
    for i in reversed(path):
        n = morph.nodes[i]
        w = n.kind if n.kind != "joint" else f"joint {n.axis}"
        notes = []
        if n.parent != 0 and n.slot:
            notes.append(ELBOW_SLOTS[n.slot])
        if n.rot:
            notes.append(f"rolled {n.rot * 90} deg")
        if notes:
            w += f" ({', '.join(notes)})"
        if n.parent == 0:
            words.append(f"core {CORE_SLOTS[n.slot]}")
        words.append(w)
    return " > ".join(words)


def servo_signs(morph, layout):
    """Mounting sign of every servo: -1 for flipped joints and right side wheels."""
    frames = morph.frames()
    out = []
    for act in layout:
        n = morph.nodes[act.slot]
        if act.kind == "joint":
            out.append(-1 if n.flip else 1)
        else:
            out.append(1 if frames[act.slot][0][1] >= 0 else -1)
    return out


def _write_assembly(morph, ctl, layout, out_dir):
    lines = [f"# Assembly: {ctl['robot']}", "",
             "Set each servo ID on the bus before mounting (one servo at a time).", "",
             "| Servo ID | Node slot | Position | Axis | Link | Sign |", "|---|---|---|---|---|---|"]
    for s, act in zip(ctl["servos"], layout):
        n = morph.nodes[act.slot]
        link = f"{_mm(n.length)} mm" if act.kind == "joint" else "-"
        lines.append(f"| {s['id']} | {act.slot} | {parent_chain(morph, act.slot)} | "
                     f"{act.axis} | {link} | {s['sign']} |")
    lines += ["",
              "## Mount orientation", "",
              "Each joint turns about one axis of its own module: pitch about the module's",
              "Z axis, yaw about its Y axis and roll about X, the direction of the link.",
              "Sign 1 follows the axis, sign -1 reverses it (the mirrored copy of a limb",
              "uses -1 where the original uses 1). A positive control value then turns the",
              "joint by the right hand rule about that axis, and wheels roll forward.",
              "If a joint moves the wrong way, flip its sign in control.json.", "",
              "## Calibration", "",
              "control.json ships with zero = 2048 for every position servo. With the robot",
              "in its straight rest pose (every link in line with the slot it hangs from),",
              "read each servo position in the firmware calibration mode and write it to",
              "`zero`. See docs/formato-controle.md.", ""]
    with open(os.path.join(out_dir, "assembly.md"), "w") as f:
        f.write("\n".join(lines))


def _write_stls(morph, out_dir):
    from darwinbot import parts
    d = os.path.join(out_dir, "parts")
    os.makedirs(d, exist_ok=True)
    out = lambda name, part: parts.export_stl(part, os.path.join(d, name))
    links, wheels, feet, elbows = part_counts(morph)
    out("torso_plate.stl", parts.torso_plate(morph.body_length * 1000, morph.body_width * 1000,
                                             mount_positions_mm(morph)))
    for length in sorted(links):
        out(f"link_{length}.stl", parts.link(length))
    out("servo_bracket.stl", parts.servo_bracket())
    if elbows:
        out("elbow_plate.stl", parts.elbow_plate())
    if feet:
        out("foot.stl", parts.foot())
    for r in sorted(wheels):
        out(f"wheel_{r}.stl", parts.wheel(r))


def export_robot(elite, out_dir, name, stl=True) -> dict:
    """Write the build folder for elite into out_dir; returns a small summary."""
    morph = morpho.decode(elite.morph)
    os.makedirs(out_dir, exist_ok=True)
    flat = get_terrains("flat")[0]
    xml, layout = builder.build_mjcf(morph, flat.assets, flat.world)
    with open(os.path.join(out_dir, "robot.xml"), "w") as f:
        f.write(xml)

    ctl = make_control_file(name, elite.fitness, layout, Controller(elite.ctrl, layout).params,
                            builder.JOINT_RANGE, signs=servo_signs(morph, layout))
    write_control_file(ctl, os.path.join(out_dir, "control.json"))

    rows = bom_rows(morph)
    _write_bom(rows, out_dir, name)
    _write_assembly(morph, ctl, layout, out_dir)
    if stl:
        _write_stls(morph, out_dir)
    return {"servos": morph.n_servos, "printed_parts": printed_parts(rows)}

"""
export.py
---------
Evolved robot -> build folder for the real DarwinBot: control.json, STL parts, bill of materials, assembly
notes and the MJCF used in simulation.

The body is a module tree (darwinbot/morphology.py), so everything is read by
walking the tree: one STL per distinct part (torso plate or its tiles, one per
link size, elbow size, wheel radius, caster wheel radius, foot radius and
spring flexure, servo bracket, splice plates), and one assembly row per servo
in layout order. Mount
positions on the plate are the core slot offsets of Morphology.slot_frame, in mm.
"""

import csv
import os
from collections import Counter

import numpy as np

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
    for slot, child in sorted(morph.child_map().get(0, {}).items()):
        off, _ = morph.slot_frame(morph.nodes[0], slot, morph.nodes[child])
        out.append((float(off[0]) * 1000, float(off[1]) * 1000))
    return out


def part_counts(morph):
    """Printed parts needed, keyed by whole millimetres: links by
    (length, width, thick), elbow plates by (distance, angle deg, width,
    thick) of the parent link, feet by radius, wheels by radius, springs
    by (length, width, blade), tails by length and caster wheels by radius;
    skids and spikes as plain counts."""
    cm = morph.child_map()
    links, wheels = Counter(), Counter()
    feet, elbows, springs, tails = Counter(), Counter(), Counter(), Counter()
    casters = Counter()
    skids = spikes = 0
    for i, n in morph.nodes.items():
        if n.kind in ("joint", "link", "spring"):
            links[(_mm(n.length), _mm(n.width), _mm(morpho.plate_thick(n)))] += 1
            if n.kind == "spring":
                springs[(_mm(n.length), _mm(n.width), _mm(n.thick))] += 1
            for slot, child in cm.get(i, {}).items():
                if slot:
                    c = morph.nodes[child]
                    elbows[(_mm(morpho.elbow_distance(c)),
                            int(round(np.degrees(morpho.elbow_angle(c)))),
                            _mm(n.width), _mm(morpho.plate_thick(n)))] += 1
            if i not in cm:
                feet[_mm(morpho.section_radius(n))] += 1   # implicit tip foot
        elif n.kind == "wheel":
            wheels[_mm(n.wheel_rad)] += 1
        elif n.kind == "caster":
            casters[_mm(n.wheel_rad)] += 1
        elif n.kind == "foot":
            feet[_mm(n.foot_rad)] += 1
        elif n.kind == "tail":
            tails[_mm(n.length)] += 1
        elif n.kind == "skid":
            skids += 1
        elif n.kind == "spike":
            spikes += 1
    return (links, wheels, feet, elbows, springs, tails, skids, spikes,
            casters)


def _torso_names(morph):
    """STL names of the torso pieces, plus (span mm, count) for each splice."""
    from darwinbot import parts
    _, _, tw, th, nx, ny = parts.tile_plan(
        morph.body_length * 1000, morph.body_width * 1000,
        mount_positions_mm(morph))
    if nx == 1 and ny == 1:
        return ["torso_plate.stl"], Counter()
    spans = Counter()
    if nx > 1:
        spans[int(round(th))] += (nx - 1) * ny
    if ny > 1:
        spans[int(round(tw))] += (ny - 1) * nx
    return [f"torso_tile_{i + 1}.stl" for i in range(nx * ny)], spans


def bom_rows(morph):
    """Bill of materials as (item, quantity, note) rows."""
    n = morph.n_servos
    (links, wheels, feet, elbows, springs, tails, skids, spikes,
     casters) = part_counts(morph)
    torso, splices = _torso_names(morph)
    rows = [("Feetech STS3215 servo, 7.4 V", n, "bus servo, ID 1..%d" % n)]
    rows += [(name, 1, "PLA") for name in torso]
    for (length, width, thick), q in sorted(links.items()):
        rows.append((f"link_{length}x{width}x{thick}.stl", q, "PLA"))
    for (length, width, blade), q in sorted(springs.items()):
        rows.append((f"spring_{width}x{blade}.stl", q, "TPU, flexure"))
    rows.append(("servo_bracket.stl", n, "PLA, one per servo"))
    for (dist, angle, width, thick), q in sorted(elbows.items()):
        rows.append((f"elbow_{dist}mm_{angle}deg_{width}x{thick}.stl", q,
                     "PLA, one per side module"))
    for radius, q in sorted(feet.items()):
        rows.append((f"foot_{radius}.stl", q, "TPU"))
    for r, q in sorted(wheels.items()):
        rows.append((f"wheel_{r}.stl", q, "PLA"))
        rows.append((f"Rubber O-ring tyre, wheel_{r}", q, f"fits rim of radius {r} mm"))
    for r, q in sorted(casters.items()):
        rows.append((f"caster_wheel_{r}.stl", q, "PLA"))
        rows.append((f"Rubber O-ring tyre, caster_wheel_{r}", q,
                     f"fits rim of radius {r} mm"))
    if casters:
        q = sum(casters.values())
        rows.append(("caster_fork.stl", q, "PLA, one per caster"))
        rows.append(("M3 x 20 bolt as caster axle", q, "one per caster"))
    for length, q in sorted(tails.items()):
        rows.append((f"tail_{length}.stl", q, "PLA"))
    if tails:
        grams = 1000.0 * sum(n.tail_mass for n in morph.nodes.values()
                              if n.kind == "tail")
        rows.append(("M3 nuts as counterweight", sum(tails.values()),
                     f"{grams:.1f} g total, one set per tail"))
    if skids:
        rows.append(("skid.stl", skids, "PLA, low-friction pad"))
    if spikes:
        rows.append(("spike.stl", spikes, "TPU, high-friction tip"))
    for span, q in sorted(splices.items()):
        rows.append((f"splice_plate_{span}.stl", q, "PLA, across a torso cut"))
    rows += [("ESP32 servo driver board", 1, "see docs/hardware-dados.md"),
             ("2S LiPo battery 7.4 V", 1, "1000 to 1500 mAh"),
             ("Step-down regulator", 1, "from 7.4 V to the board supply"),
             ("M3 screws",
              4 * n + 4 * sum(elbows.values()) + 4 * sum(splices.values()) + 4,
              "estimate: 4 per servo, 4 per elbow or splice plate, 4 for the board")]
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
        w = f"{n.kind} {n.axis}" if n.kind in ("joint", "spring") else n.kind
        notes = []
        if n.parent != 0 and n.slot:
            notes.append(ELBOW_SLOTS[n.slot])
        if n.rot:
            notes.append(f"rolled {np.degrees(n.rot):.0f} deg")
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
    links, wheels, feet, elbows, springs, tails, skids, spikes, \
        casters = part_counts(morph)
    torso, splices = _torso_names(morph)
    tiles = parts.torso_tiles(morph.body_length * 1000, morph.body_width * 1000,
                              mount_positions_mm(morph))
    for name, tile in zip(torso, tiles):
        out(name, tile)
    for span in sorted(splices):
        out(f"splice_plate_{span}.stl", parts.splice_plate(span))
    for length, width, thick in sorted(links):
        out(f"link_{length}x{width}x{thick}.stl", parts.link(length, width, thick))
    for length, width, blade in sorted(springs):
        out(f"spring_{width}x{blade}.stl", parts.spring(length, width, blade))
    out("servo_bracket.stl", parts.servo_bracket())
    for dist, angle, width, thick in sorted(elbows):
        out(f"elbow_{dist}mm_{angle}deg_{width}x{thick}.stl",
            parts.elbow_plate(dist, angle, width, thick))
    for radius in sorted(feet):
        out(f"foot_{radius}.stl", parts.foot(radius))
    for r in sorted(wheels):
        out(f"wheel_{r}.stl", parts.wheel(r))
    for r in sorted(casters):
        out(f"caster_wheel_{r}.stl", parts.caster_wheel(r))
    if casters:
        out("caster_fork.stl", parts.caster_fork())
    for length in sorted(tails):
        out(f"tail_{length}.stl", parts.tail_bar(length))
    if skids:
        out("skid.stl", parts.skid())
    if spikes:
        out("spike.stl", parts.spike())


def export_robot(elite, out_dir, name, stl=True, payload=0.0) -> dict:
    """Write the build folder for elite into out_dir; returns a small summary."""
    morph = morpho.decode(elite.morph)
    os.makedirs(out_dir, exist_ok=True)
    flat = get_terrains("flat")[0]
    xml, layout = builder.build_mjcf(morph, flat.assets, flat.world,
                                    payload=payload)
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

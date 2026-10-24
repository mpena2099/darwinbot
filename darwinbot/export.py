"""
export.py
---------
Evolved robot -> build folder for the real DarwinBot (Phase 4 of
docs/plano-hardware.md): control.json, STL parts, bill of materials, assembly
notes and the MJCF used in simulation.

Hip positions on the printed torso plate come from the same formulas as
builder._leg_xml, converted to mm.
"""

import csv
import os
from collections import Counter

from darwinbot import builder
from darwinbot import morphology as morpho
from darwinbot.control_file import make_control_file, write_control_file
from darwinbot.controller import CPG
from darwinbot.terrain import get_terrains

JOINT_WORDS = {"hip_yaw": "swing", "hip_pitch": "lift", "knee": "knee bend",
               "wheel": "wheel drive"}
JOINT_PLACE = {"hip_yaw": "hip", "hip_pitch": "hip", "knee": "knee", "wheel": "wheel"}


def hip_positions_mm(morph):
    """Hip (x, y) on the torso plate for every leg, both sides, in mm."""
    out = []
    for pair in morph.pairs:
        x = pair.attach_x * morph.body_length / 2 * 0.9 * 1000
        for sign in (1, -1):
            out.append((x, sign * morph.body_width / 2 * 1000))
    return out


def _mm(m):
    return int(round(m * 1000))


def bom_rows(morph):
    """Bill of materials as (item, quantity, note) rows."""
    n = morph.n_servos
    legs = Counter()      # link length mm -> count
    n_feet = 0
    wheels = Counter()    # wheel radius mm -> count
    for pair in morph.pairs:
        legs[_mm(pair.thigh_len)] += 2
        legs[_mm(pair.shin_len)] += 2
        if pair.has_wheel:
            wheels[_mm(pair.wheel_rad)] += 2
        else:
            n_feet += 2
    rows = [("Feetech STS3215 servo, 7.4 V", n, "bus servo, ID 1..%d" % n),
            ("torso_plate.stl", 1, "PLA")]
    for length, q in sorted(legs.items()):
        rows.append((f"link_{length}.stl", q, "PLA"))
    rows.append(("servo_bracket.stl", n, "PLA, one per servo"))
    if n_feet:
        rows.append(("foot.stl", n_feet, "TPU"))
    for r, q in sorted(wheels.items()):
        rows.append((f"wheel_{r}.stl", q, "PLA"))
        rows.append((f"Rubber O-ring tyre, wheel_{r}", q, f"fits rim of radius {r} mm"))
    rows += [("ESP32 servo driver board", 1, "see docs/hardware-dados.md"),
             ("2S LiPo battery 7.4 V", 1, "1000 to 1500 mAh"),
             ("Step-down regulator", 1, "from 7.4 V to the board supply"),
             ("M3 screws", 4 * n + 4, "estimate: 4 per servo + 4 per board")]
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


def _leg_place(pair, side, kind):
    where = "front" if pair.attach_x > 0.33 else "rear" if pair.attach_x < -0.33 else "middle"
    return f"{where} {'left' if side == 0 else 'right'} {JOINT_PLACE[kind]}, {JOINT_WORDS[kind]}"


def _write_assembly(morph, ctl, layout, out_dir):
    lines = [f"# Assembly: {ctl['robot']}", "",
             "Set each servo ID on the bus before mounting (one servo at a time).", "",
             "| Servo ID | Joint | Position | Sign |", "|---|---|---|---|"]
    for s, (pi, side, kind) in zip(ctl["servos"], layout):
        pair = next(p for p in morph.pairs if p.index == pi)
        lines.append(f"| {s['id']} | {s['name']} | {_leg_place(pair, side, kind)} | {s['sign']} |")
    lines += ["",
              "## Mount orientation", "",
              "Left and right servos are mirrored. Sign 1 is the left side, sign -1 the right.",
              "Mount the horn so that positive control values lift the thigh, bend the knee",
              "outward, swing the hip yaw forward and roll the wheel forward.",
              "If a joint moves the wrong way, flip its sign in control.json.", "",
              "## Calibration", "",
              "control.json ships with zero = 2048 for every position servo. With the leg",
              "straight out (thigh horizontal, shin vertical, wheel axle along Y), read each",
              "servo position in the firmware calibration mode and write it to `zero`.",
              "See docs/formato-controle.md.", ""]
    with open(os.path.join(out_dir, "assembly.md"), "w") as f:
        f.write("\n".join(lines))


def _write_stls(morph, out_dir):
    from darwinbot import parts
    d = os.path.join(out_dir, "parts")
    os.makedirs(d, exist_ok=True)
    out = lambda name, part: parts.export_stl(part, os.path.join(d, name))
    out("torso_plate.stl", parts.torso_plate(morph.body_length * 1000, morph.body_width * 1000,
                                             hip_positions_mm(morph)))
    lengths = {_mm(p.thigh_len) for p in morph.pairs} | {_mm(p.shin_len) for p in morph.pairs}
    for length in sorted(lengths):
        out(f"link_{length}.stl", parts.link(length))
    out("servo_bracket.stl", parts.servo_bracket())
    if any(not p.has_wheel for p in morph.pairs):
        out("foot.stl", parts.foot())
    for r in sorted({_mm(p.wheel_rad) for p in morph.pairs if p.has_wheel}):
        out(f"wheel_{r}.stl", parts.wheel(r))


def export_robot(elite, out_dir, name, stl=True) -> dict:
    """Write the build folder for elite into out_dir; returns a small summary."""
    morph = morpho.decode(elite.morph)
    os.makedirs(out_dir, exist_ok=True)
    flat = get_terrains("flat")[0]
    xml, layout = builder.build_mjcf(morph, flat.assets, flat.world)
    with open(os.path.join(out_dir, "robot.xml"), "w") as f:
        f.write(xml)

    ctl = make_control_file(name, elite.fitness, layout, CPG(elite.ctrl, layout),
                            builder.JOINT_RANGE)
    write_control_file(ctl, os.path.join(out_dir, "control.json"))

    rows = bom_rows(morph)
    _write_bom(rows, out_dir, name)
    _write_assembly(morph, ctl, layout, out_dir)
    if stl:
        _write_stls(morph, out_dir)
    return {"servos": morph.n_servos, "printed_parts": printed_parts(rows)}

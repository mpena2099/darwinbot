"""
export.py
---------
Evolved robot -> build folder for the real DarwinBot: control.json, STL parts, bill of materials, assembly
notes and the MJCF used in simulation.

The body is a module tree (darwinbot/morphology.py), so everything is read by
walking the tree: one STL per distinct part (the extruded torso contour plate
or its tiles, one per link size and profile, elbow size, wheel radius, caster
wheel radius, foot shape and radius and spring flexure, servo bracket, splice
plates, one plate per spine segment size), and one assembly row per servo in
layout order. Mount positions on the plate are the core slot offsets of
Morphology.slot_frame, in mm.
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
              "right-front", "right-middle", "right-rear",
              "top front", "top rear", "bottom front", "bottom rear")
ELBOW_SLOTS = {1: "left elbow", 2: "right elbow"}


def _mm(m):
    return int(round(m * 1000))


def link_name(profile, bend_deg, length, width, thick):
    """STL name of a link, with the profile and whole-degree bend when bent."""
    name = f"link_{length}x{width}x{thick}"
    if profile != "straight":
        name += f"_{profile}{int(round(bend_deg))}"
    return name


def foot_name(shape, radius):
    """STL name of a foot: the sphere keeps its old name."""
    return f"foot_{radius}" if shape == "sphere" else f"foot_{shape}_{radius}"


# Stack-up constants, equal to parts.WALL, parts.PLATE_T (parts needs build123d,
# the bill of materials does not).
FLOOR_T = 2.5            # servo bracket floor
PLATE_T = 6.0            # torso plate and splice strip
NUT_T = 2.4              # M3 nut
M3_LENGTHS = (6, 8, 10, 12, 16, 20, 25, 30)


def screw_length(stack_mm):
    """Shortest stock M3 screw that crosses the stack and reaches a nut."""
    need = stack_mm + NUT_T
    return next((L for L in M3_LENGTHS if L >= need), M3_LENGTHS[-1])


def mount_positions_mm(morph):
    """(x, y, yaw in degrees) on the torso plate of every used core slot, in
    mm. The yaw is the attachment angle of the child on the core, in the
    frame of the slot; it turns the tab footprint and the mounting pattern.
    A face mount (top or bottom) has the yaw about the vertical and a fourth
    element True: it is drilled through the plate, with no edge tab."""
    out = []
    for slot, child in sorted(morph.child_map().get(0, {}).items()):
        off, rot = morph.slot_frame(morph.nodes[0], slot, morph.nodes[child])
        x, y = float(off[0]) * 1000, float(off[1]) * 1000
        if slot in morpho.FACE_SLOTS:      # local X is vertical: read the yaw off Y
            out.append((x, y, float(np.degrees(np.arctan2(-rot[0, 1], rot[1, 1]))),
                        True))
        else:
            out.append((x, y, float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))))
    return out


def segment_mounts_mm(morph, seg_id):
    """(x, y, yaw in degrees) mounts on the plate of one segment, in mm and in
    the frame of the plate: the spine servo at its near end, then every used
    slot. Slot offsets of Morphology.slot_frame minus the plate centre."""
    node = morph.nodes[seg_id]
    centre = morpho.segment_center(node)
    out = [(-node.length * 500.0, 0.0, 180.0)]
    for slot, child in sorted(morph.child_map().get(seg_id, {}).items()):
        off, rot = morph.slot_frame(node, slot, morph.nodes[child])
        out.append(((float(off[0]) - centre[0]) * 1000,
                    (float(off[1]) - centre[1]) * 1000,
                    float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))))
    return out


def segment_parts(morph):
    """{(length mm, width mm): mounts} of the spine plates, one entry per size;
    equal sizes share one STL carrying the mounts of all of them."""
    out = {}
    for i, n in morph.nodes.items():
        if n.kind == "segment":
            mounts = out.setdefault((_mm(n.length), _mm(n.width)), [])
            for m in segment_mounts_mm(morph, i):
                if all(max(abs(a - b) for a, b in zip(m, k)) > 0.5
                       for k in mounts):
                    mounts.append(m)
    return out


def part_counts(morph):
    """Printed parts needed, keyed by whole millimetres: links by
    (length, width, thick, profile, bend deg), elbow plates by (distance,
    angle deg, width, thick) of the parent link, feet by (shape, radius),
    wheels by radius, springs by (width, blade), tails by length and caster
    wheels by radius; skids and spikes as plain counts."""
    cm = morph.child_map()
    links, wheels = Counter(), Counter()
    feet, elbows, springs, tails = Counter(), Counter(), Counter(), Counter()
    casters = Counter()
    skids = spikes = 0
    for i, n in morph.nodes.items():
        if n.kind in ("joint", "link", "spring"):
            bend = int(round(np.degrees(n.bend))) if n.profile != "straight" else 0
            links[(_mm(n.length), _mm(n.width), _mm(morpho.plate_thick(n)),
                   n.profile, bend)] += 1
            if n.kind == "spring":
                springs[(_mm(n.width), _mm(n.thick))] += 1
            for slot, child in cm.get(i, {}).items():
                if slot:
                    c = morph.nodes[child]
                    elbows[(_mm(morpho.elbow_distance(c)),
                            int(round(np.degrees(morpho.elbow_angle(c)))),
                            _mm(n.width), _mm(morpho.plate_thick(n)))] += 1
            if i not in cm:
                feet[("sphere", _mm(morpho.section_radius(n)))] += 1
        elif n.kind == "wheel":
            wheels[_mm(n.wheel_rad)] += 1
        elif n.kind == "caster":
            casters[_mm(n.wheel_rad)] += 1
        elif n.kind == "foot":
            feet[(n.foot_shape, _mm(n.foot_rad))] += 1
        elif n.kind == "tail":
            tails[_mm(n.length)] += 1
        elif n.kind == "skid":
            skids += 1
        elif n.kind == "spike":
            spikes += 1
    return (links, wheels, feet, elbows, springs, tails, skids, spikes,
            casters)


def torso_polygon_mm(morph):
    """Torso contour vertices in mm, for the printable plate."""
    return [(x * 1000, y * 1000) for x, y in morph.torso_polygon()]


def torso_size_mm(morph):
    """Bounding box (x, y) of the torso contour in mm."""
    poly = torso_polygon_mm(morph)
    return (max(x for x, _ in poly) - min(x for x, _ in poly),
            max(y for _, y in poly) - min(y for _, y in poly))


def _torso_names(morph):
    """STL names of the torso pieces, plus (span mm, count) for each splice."""
    from darwinbot import parts
    _, _, tw, th, nx, ny = parts.tile_plan(
        torso_polygon_mm(morph), mount_positions_mm(morph))
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
    size = torso_size_mm(morph)
    rows += [(name, 1, f"PLA, {size[0]:.0f} x {size[1]:.0f} mm") for name in torso]
    sizes = Counter((_mm(s.length), _mm(s.width))
                    for s in morph.nodes.values() if s.kind == "segment")
    for (length, width), q in sorted(sizes.items()):
        rows.append((f"segment_plate_{length}x{width}.stl", q,
                     "PLA, second torso plate behind a spine servo"))
    for (length, width, thick, profile, bend), q in sorted(links.items()):
        rows.append((f"{link_name(profile, bend, length, width, thick)}.stl",
                     q, "PLA"))
    for (width, blade), q in sorted(springs.items()):
        rows.append((f"spring_{width}x{blade}.stl", q, "TPU, flexure"))
    rows.append(("servo_bracket.stl", n, "PLA, one per servo"))
    for (dist, angle, width, thick), q in sorted(elbows.items()):
        rows.append((f"elbow_{dist}mm_{angle}deg_{width}x{thick}.stl", q,
                     "PLA, one per side module"))
    for (shape, radius), q in sorted(feet.items()):
        rows.append((f"{foot_name(shape, radius)}.stl", q, "TPU"))
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
    screws = screw_counts(morph, elbows, springs, splices)
    rows.append(("ESP32 servo driver board", 1,
                 "see docs/hardware-dados.md; the torso hole pattern is a "
                 "PLACEHOLDER, check it against your board before printing"))
    rows += [("2S LiPo battery 7.4 V", 1, "1000 to 1500 mAh"),
             ("Step-down regulator", 1, "from 7.4 V to the board supply")]
    for length, (q, what) in sorted(screws.items()):
        rows.append((f"M3 x {length} screw", q, what))
    rows.append(("M3 nut", sum(q for q, _ in screws.values()),
                 "one per screw"))
    return rows


def screw_counts(morph, elbows, springs, splices):
    """{M3 screw length: (count, where used)}, from the stack-up at each joint:
    every servo bracket floor bolts to a torso tab or a link end (or elbow
    plate), each elbow plate bolts on a link end, each spring has two blocks
    bolted to their neighbours, splices join two tiles, and the board sits on
    the torso. Four screws per interface."""
    use = {}
    def add(stack, n, what):
        L = screw_length(stack)
        q, w = use.get(L, (0, []))
        use[L] = (q + n, w + [what] if what not in w else w)
    for n in morph.nodes.values():
        if morpho.is_joint(n.kind) or n.kind == "wheel":
            if morph.nodes[n.parent].kind in ("core", "segment"):
                add(PLATE_T + FLOOR_T, 4, "bracket on torso")
            else:
                add(_mm(morpho.plate_thick(morph.nodes[n.parent])) + FLOOR_T,
                    4, "bracket on link")
    for (dist, angle, width, thick), q in elbows.items():
        add(2 * thick, 4 * q, "elbow on link")
    for (width, blade), q in springs.items():
        add(2 * max(6, blade), 8 * q, "spring blocks")
    for span, q in splices.items():
        add(2 * PLATE_T, 4 * q, "torso splice")
    add(PLATE_T, 4, "ESP32 board")
    return {L: (q, ", ".join(w)) for L, (q, w) in use.items()}


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
        if n.kind == "segment":
            w = f"spine servo {n.axis}"
        on_torso = morph.nodes[n.parent].kind in ("core", "segment")
        notes = []
        if not on_torso and n.slot:
            notes.append(ELBOW_SLOTS[n.slot])
        if n.rot:
            notes.append(f"rolled {np.degrees(n.rot):.0f} deg")
        if notes:
            w += f" ({', '.join(notes)})"
        if on_torso:
            base = "core" if n.parent == 0 else "spine plate"
            words.append(f"{base} {CORE_SLOTS[n.slot]}")
        words.append(w)
    return " > ".join(words)


def servo_signs(morph, layout):
    """Mounting sign of every servo: -1 for flipped joints. A wheel servo turns
    its horn, which sits on the hub side of the servo; positive then means
    about the hub direction. The simulation spins the wheel about the local
    axle sign morph.wheel_axis_sign, so the mounting sign is the product of
    that axle sign and the side of the hub."""
    frames = morph.frames()
    out = []
    for act in layout:
        n = morph.nodes[act.slot]
        if act.kind == "joint":
            out.append(-1 if n.flip else 1)
        else:
            pos, rot = frames[act.slot]
            hub = morph.wheel_hub(n, pos, rot)
            side = 1 if hub[1] >= 0 else -1
            out.append(int(morph.wheel_axis_sign(rot)) * side)
    return out


def _write_assembly(morph, ctl, layout, out_dir):
    w, h = torso_size_mm(morph)
    lines = [f"# Assembly: {ctl['robot']}", "",
             f"Torso plate bounding box: {w:.0f} x {h:.0f} mm.", "",
             "Set each servo ID on the bus before mounting (one servo at a time).", "",
             "| Servo ID | Node slot | Position | Axis | Link | Sign |", "|---|---|---|---|---|---|"]
    for s, act in zip(ctl["servos"], layout):
        n = morph.nodes[act.slot]
        link = (f"{_mm(n.length)} mm"
                if act.kind == "joint" and n.kind == "joint" else "-")
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


def _write_organic(morph, out_dir) -> dict:
    """Hollowed links and elbow plates in parts_organic/ (organic.hollow);
    returns {file name: volume ratio}."""
    from darwinbot import organic, parts
    d = os.path.join(out_dir, "parts_organic")
    os.makedirs(d, exist_ok=True)
    links, _, _, elbows = part_counts(morph)[:4]
    ratios = {}
    for length, width, thick, profile, bend in sorted(links):
        name = f"{link_name(profile, bend, length, width, thick)}.stl"
        tip = parts._link_path_mm(length, profile, bend)[-1]
        part, ratios[name] = organic.hollow(parts.link(length, width, thick, profile, bend),
                                            (0.0, 0.0), tip, parts._boss_d(width) / 2 + 1)
        parts.export_stl(part, os.path.join(d, name))
    for dist, angle, width, thick in sorted(elbows):
        name = f"elbow_{dist}mm_{angle}deg_{width}x{thick}.stl"
        part, ratios[name] = organic.hollow(parts.elbow_plate(dist, angle, width, thick),
                                            (0.0, 0.0), (0.0, dist), parts._boss_d(width) / 2 + 1)
        parts.export_stl(part, os.path.join(d, name))
    return ratios


def _write_stls(morph, out_dir):
    from darwinbot import parts
    d = os.path.join(out_dir, "parts")
    os.makedirs(d, exist_ok=True)
    out = lambda name, part: parts.export_stl(part, os.path.join(d, name))
    links, wheels, feet, elbows, springs, tails, skids, spikes, \
        casters = part_counts(morph)
    torso, splices = _torso_names(morph)
    tiles = parts.torso_tiles(torso_polygon_mm(morph), mount_positions_mm(morph))
    for name, tile in zip(torso, tiles):
        out(name, tile)
    for (length, width), mounts in sorted(segment_parts(morph).items()):
        half = [(length / 2, 0), (length / 2, width / 2), (0, width / 2),
                (-length / 2, width / 2), (-length / 2, 0),
                (-length / 2, -width / 2), (0, -width / 2),
                (length / 2, -width / 2)]
        out(f"segment_plate_{length}x{width}.stl",
            parts.torso_plate(half, mounts, electronics=False))
    for span in sorted(splices):
        out(f"splice_plate_{span}.stl", parts.splice_plate(span))
    for length, width, thick, profile, bend in sorted(links):
        out(f"{link_name(profile, bend, length, width, thick)}.stl",
            parts.link(length, width, thick, profile, bend))
    for width, blade in sorted(springs):
        out(f"spring_{width}x{blade}.stl", parts.spring(width, blade))
    out("servo_bracket.stl", parts.servo_bracket())
    for dist, angle, width, thick in sorted(elbows):
        out(f"elbow_{dist}mm_{angle}deg_{width}x{thick}.stl",
            parts.elbow_plate(dist, angle, width, thick))
    for shape, radius in sorted(feet):
        out(f"{foot_name(shape, radius)}.stl", parts.foot(radius, shape))
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


def export_robot(elite, out_dir, name, stl=True, payload=0.0, organic=False) -> dict:
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
    info = {"servos": morph.n_servos, "printed_parts": printed_parts(rows)}
    if stl and organic:
        info["organic"] = _write_organic(morph, out_dir)
    return info

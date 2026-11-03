"""
assembly.py
-----------
Where every printed part of a robot sits in the spawn pose: one record per
part with its STL name and make function, the module it belongs to, the
simulation body its frame comes from, and its position (mm) and rotation in
that body's frame. render_assembled.py puts the meshes there and
check_assembly.py builds the solids to check that they fit.

The frames are the simulation's: a module frame is the servo body centre and a
link plate is centred on the joint plane, so the printed stack and the
simulated one differ by the servo's real output-axis offset. A part is owned by
the module whose frame it is drawn in, except an elbow plate, which belongs to
the child module whose servo it carries.
"""

from typing import Callable, NamedTuple

import numpy as np

from darwinbot import export, parts
from darwinbot import morphology as mo

IDENTITY = np.eye(3)
PLA, TPU, SPRING = "pla", "tpu", "spring"


class Placement(NamedTuple):
    label:  str                # unique name, for the report
    owner:  int                # module id the part belongs to (-1: the torso)
    body:   str                # simulation body the frame comes from
    pos:    tuple              # mm, in the body frame
    rot:    np.ndarray         # 3x3, in the body frame
    name:   str                # STL name
    make:   Callable           # () -> build123d Part
    kind:   str                # printed material: PLA, TPU or SPRING
    horn:   bool = False       # bolted to a servo horn: has the horn holes


def rx(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def placements(morph) -> list:
    """One Placement per printed part of `morph`; equal sizes are not merged."""
    out = []
    cm = morph.child_map()
    sz = np.array(mo.SERVO_BOX) * 500.0         # bracket around the lower 60%
    out.append(Placement(
        "torso_plate", -1, "torso", (0, 0, -(sz[2] + parts.WALL + parts.PLATE_T)),
        IDENTITY, "torso_plate",
        lambda: parts.torso_plate(export.torso_polygon_mm(morph),
                                  export.mount_positions_mm(morph)), PLA))
    for i, node in morph.nodes.items():
        body = f"n{i}"
        if node.kind in ("joint", "wheel", "segment"):
            out.append(Placement(f"bracket{i}", i, body,
                                 (0, 0, -(sz[2] + parts.WALL)), IDENTITY,
                                 "servo_bracket", parts.servo_bracket, PLA))
        if node.kind in ("joint", "link", "spring"):
            t = export._mm(mo.plate_thick(node))
            L, W = export._mm(node.length), export._mm(node.width)
            bend = int(round(np.degrees(node.bend))) if node.profile != "straight" else 0
            out.append(Placement(
                f"link{i}", i, body, (0, 0, -t / 2), IDENTITY,
                export.link_name(node.profile, bend, L, W, t),
                lambda L=L, W=W, t=t, p=node.profile, b=bend: parts.link(L, W, t, p, b),
                SPRING if node.kind == "spring" else PLA, horn=node.kind != "spring"))
            tip, rot = mo.link_tip(node)
            for slot, c in sorted(cm.get(i, {}).items()):
                if not slot:
                    continue
                d = export._mm(mo.elbow_distance(morph.nodes[c]))
                ang = int(round(np.degrees(mo.elbow_angle(morph.nodes[c]))))
                r = rot @ (IDENTITY if slot == 1 else rx(180))
                out.append(Placement(
                    f"elbow{i}_{c}", c, body, tuple(tip * 1000 + r @ [0, 0, -t / 2]), r,
                    f"elbow_{d}mm_{ang}deg_{W}x{t}",
                    lambda d=d, a=ang, W=W, t=t: parts.elbow_plate(d, a, W, t), PLA))
            if i not in cm and node.kind != "spring":     # free end: TPU cap
                r_mm = export._mm(mo.section_radius(node))
                out.append(Placement(f"foot{i}", i, body, tuple(tip * 1000), rot @ ry(90),
                                     f"foot_{r_mm}", lambda r=r_mm: parts.foot(r), TPU))
        elif node.kind == "foot":
            r_mm = export._mm(node.foot_rad)
            out.append(Placement(f"foot{i}", i, body, (0, 0, 0),
                                 ry(90) if node.foot_shape == "sphere" else IDENTITY,
                                 export.foot_name(node.foot_shape, r_mm),
                                 lambda r=r_mm, s=node.foot_shape: parts.foot(r, s), TPU))
        elif node.kind == "wheel":
            r_mm = export._mm(node.wheel_rad)
            out.append(Placement(f"wheel{i}", i, f"w{i}", (0, 0, 0), rx(90),
                                 f"wheel_{r_mm}", lambda r=r_mm: parts.wheel(r), PLA))
    return out

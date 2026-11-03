"""
assembly.py
-----------
Where every printed part of a robot sits in a pose: one record per part with
its STL name and make function, the module it belongs to, the simulation body
its frame comes from, and its position (mm) and rotation in that body's frame.
render_assembled.py puts the meshes there and check_assembly.py builds the
solids to check that they fit.

The frames are the simulation's: a module frame is the servo body centre and a
link plate is centred on the joint plane. A cradle belongs to the parent of
its servo, at the joint frame in the parent's body; a U link belongs to the
child, in its own body. A cradle foot is turned about the joint axis so it
reaches down to the torso plate whichever way the slot is rolled.
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


def flip_rot(axis: str) -> np.ndarray:
    """Turn of 180 degrees about the cradle's own X (the servo length), in the
    module frame of a joint about `axis`: it carries the foot from one side of
    the cradle to the other, and maps the cradle onto itself."""
    r = export.CRADLE_FRAME[axis]
    return r @ np.diag([1.0, -1.0, -1.0]) @ r.T


def wedge_rot(rel: np.ndarray, side: int, axis: str = "pitch") -> np.ndarray:
    """Rotation of a wedge on a torso tab: its bottom face on the plate (the
    torso body's Z), its top rising along the horizontal part of the rolled
    mount's foot, so the tilted top stays parallel to the cradle foot."""
    u = rel @ export.CRADLE_FRAME[axis] @ np.array([0.0, -side, 0.0])
    h = np.array([u[0], u[1], 0.0])
    h /= np.linalg.norm(h)
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack([h, np.cross(z, h), z])


def printed_rest(morph, node):
    """(offset mm, rotation) of `node` in its parent's frame at rest, as the
    printed parts put it: a link is printed in whole millimetres and its bend
    in whole degrees, an elbow distance, angle and roll the same, so the tip a
    child hangs on is where the printed path ends and not where the simulation
    has it (half a millimetre or degree off). On the torso the simulation's
    slot is kept."""
    parent = morph.nodes[node.parent]
    off, rot = morph.rel_frame(node)
    if parent.kind not in ("joint", "link", "spring"):
        return off * 1000.0, rot
    L = export._mm(parent.length)
    bend = int(round(np.degrees(parent.bend))) if parent.profile != "straight" else 0
    tip = np.array([*parts._link_path_mm(L, parent.profile, bend)[-1], 0.0])
    trot = rz(bend if parent.profile in ("L", "arc") else 0.0)
    roll = rx(int(round(np.degrees(node.rot))))
    if not node.slot:
        return tip, trot @ roll
    side = 1.0 if node.slot == 1 else -1.0
    d = export._mm(mo.elbow_distance(node))
    a = int(round(np.degrees(mo.elbow_angle(node))))
    return tip + trot @ np.array([0.0, side * d, 0.0]), trot @ rz(side * a) @ roll


def rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def printed_frames(morph, frames: dict) -> dict:
    """The body frames of a pose as the printed robot has them: the torso and
    every joint angle are the simulation's, but each module sits where the
    printed links and elbow plates put it (see printed_rest)."""
    out = {"torso": frames["torso"]}
    for i in morph.order[1:]:
        node = morph.nodes[i]
        pb = "torso" if node.parent == 0 else f"n{node.parent}"
        (pt, pr), (st, sr) = out[pb], frames[pb]
        off_sim, rot_sim = morph.rel_frame(node)
        joint = rot_sim.T @ (sr.T @ frames[f"n{i}"][1])      # the joint's turn
        off, rot = printed_rest(morph, node)
        out[f"n{i}"] = (np.asarray(pt) + pr @ off, pr @ rot @ joint)
        if f"w{i}" in frames:
            wt, wr = frames[f"w{i}"]
            nt, nr = frames[f"n{i}"]
            local = nr.T @ (np.asarray(wt) - np.asarray(nt))
            out[f"w{i}"] = (out[f"n{i}"][0] + out[f"n{i}"][1] @ local,
                            out[f"n{i}"][1] @ (nr.T @ wr))
    return out


def servo_axis(morph, i: int):
    """Module axis of the servo at node i, or None for a passive module."""
    return export.servo_axis(morph, i)


def placements(morph) -> list:
    """One Placement per printed part of `morph`; equal sizes are not merged."""
    out = []
    cm = morph.child_map()
    plate_drop = export.plate_drop(morph)
    out.append(Placement(
        "torso_plate", -1, "torso",
        (0, 0, -(plate_drop + parts.PLATE_T)), IDENTITY, "torso_plate",
        lambda: parts.torso_plate(export.torso_polygon_mm(morph),
                                  export.mount_positions_mm(morph)), PLA))
    for i, node in morph.nodes.items():
        body = f"n{i}"
        parent_body = "torso" if node.parent == 0 else f"n{node.parent}"
        axis = export.servo_axis(morph, i)
        if axis:
            off, rel = morph.rel_frame(node)
            mount = export.torso_mount(morph, i)
            if mount:
                turn = rel if mount.side > 0 else rel @ flip_rot(axis)
                out.append(Placement(
                    f"cradle{i}", i, parent_body, tuple(off * 1000.0), turn,
                    export.cradle_name(axis, mount.angle, mount.kind),
                    lambda a=axis, m=mount: parts.cradle(
                        a, m.kind != "free", 1, m.drop, m.shift,
                        back=m.kind == "back"), PLA))
                if mount.angle:
                    out.append(Placement(
                        f"wedge{i}", i, parent_body,
                        (off[0] * 1000.0, off[1] * 1000.0, -plate_drop),
                        wedge_rot(rel, mount.side, axis),
                        export.wedge_name(abs(mount.angle)),
                        lambda a=abs(mount.angle): parts.wedge(a), PLA))
        if node.kind in ("joint", "link", "spring"):
            L, W, t, profile, bend, u, foot, tip, elbow = export.link_args(
                morph, i)
            name = export.link_name(profile, bend, L, W, t, u, foot, tip, elbow)
            if export.print_owner(morph, i) == i:     # else printed with its owner
                out.append(Placement(
                    f"link{i}", i, body, (0, 0, 0), IDENTITY, name,
                    lambda a=(L, W, t, profile, bend), u=u, f=foot, x=tip, e=elbow:
                        parts.link(*a, axis=u, foot_r=f, tip_axis=x, elbow=e),
                    SPRING if node.kind == "spring" else PLA,
                    horn=node.kind == "joint"))
            # the printed link has whole millimetre measures: its tip is where
            # the printed path ends, up to half a millimetre from the simulation's
            tip_pt, rot = mo.link_tip(node)
            tip_pt = np.array([*parts._link_path_mm(L, profile, bend)[-1], 0.0]) / 1000.0
            for slot, c in sorted(cm.get(i, {}).items()):
                if not slot or not export.servo_axis(morph, c):
                    continue            # passive side modules print with the link
                d, ang, W, t, cx = export.elbow_args(morph, i, c)
                r = rot @ (IDENTITY if slot == 1 else rx(180))
                out.append(Placement(
                    f"elbow{i}_{c}", c, body, tuple(tip_pt * 1000), r,
                    export.elbow_name(d, ang, W, t, cx),
                    lambda d=d, a=ang, W=W, t=t, cx=cx: parts.elbow_plate(
                        d, a, W, t, axis=cx,
                        trim=export.elbow_trim(d, a, *cx)[0]), PLA))
            if i not in cm and node.kind != "spring":     # free end: TPU cap
                r_mm = export._mm(mo.section_radius(node))
                out.append(Placement(
                    f"foot{i}", i, body, tuple(tip_pt * 1000), rot @ ry(-90),
                    f"foot_{r_mm}", lambda r=r_mm: parts.foot(r), TPU))
        elif node.kind == "foot":
            r_mm = export._mm(node.foot_rad)
            out.append(Placement(f"foot{i}", i, body, (0, 0, 0),
                                 ry(-90) if node.foot_shape == "sphere" else IDENTITY,
                                 export.foot_name(node.foot_shape, r_mm),
                                 lambda r=r_mm, s=node.foot_shape: parts.foot(r, s), TPU))
        elif node.kind == "wheel":
            r_mm = export._mm(node.wheel_rad)
            out.append(Placement(f"wheel{i}", i, f"w{i}", (0, 0, 0), rx(90),
                                 f"wheel_{r_mm}", lambda r=r_mm: parts.wheel(r), PLA))
    return out

"""
check_assembly.py
-----------------
Check that one robot of an archive can be printed and screwed together: no two
printed parts, and no printed part and servo body, share more than a cubic
millimetre, and every link hole lines up with the servo horn holes it is
screwed to.

    python tools/check_assembly.py --archive a.pkl --type 7servos-mid-trot
    python tools/check_assembly.py --archive a.pkl --type <type> --servo-step sts3215.step

The printed parts are build123d solids in the spawn pose, at the frames
render_assembled.py puts the meshes in (tools/assembly.py). Servo bodies are
boxes with their output axis where parts.py draws the horn holes; with
`--servo-step` a STEP model replaces the box (origin at the body centre, x
along the body, z along the output axis). A part is compared only with parts
of another module, and never with those of its parent: a module frame is the
simulated servo body centre, so the plate on a horn, the bracket around the
body and the plates stacked at a link tip overlap by that simplification, a
known simulation decision and not a printing error.
"""

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from assembly import placements                             # noqa: E402

from build123d import Box, Location, Pos, gp_Trsf          # noqa: E402

from darwinbot import builder, parts                       # noqa: E402
from darwinbot import morphology as mo                     # noqa: E402

MIN_OVERLAP = 1.0    # mm^3 shared by two solids before it counts as a clash
MAX_GAP = 0.3        # mm between a link hole centre and the nearest horn hole


def spawn_frames(morph) -> dict:
    """{body: (position mm, rotation)} of the model at the spawn pose."""
    model = mujoco.MjModel.from_xml_string(builder.build_mjcf(morph)[0])
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    out = {}
    for k in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, k)
        if name:
            out[name] = (np.array(data.xpos[k]) * 1000.0,
                         np.array(data.xmat[k]).reshape(3, 3).copy())
    return out


def _moved(pos, rot, solid):
    """`solid` from its own frame to `pos` (mm) and `rot` in the world."""
    trsf = gp_Trsf()
    trsf.SetValues(*rot[0], pos[0], *rot[1], pos[1], *rot[2], pos[2])
    return solid.moved(Location(trsf))


def servo_body(step_path=None):
    """Servo body in its own frame: the output axis is +Z at x = SERVO_AXIS_X."""
    if step_path:
        from build123d import import_step
        return Pos(-parts.SERVO_AXIS_X, 0, 0) * import_step(step_path)
    return Pos(-parts.SERVO_AXIS_X, 0, parts.SERVO_Z0 + parts.SERVO_BODY_H / 2) * Box(
        parts.SERVO_L, parts.SERVO_W, parts.SERVO_BODY_H)


def _joined(a: int, b: int, morph) -> bool:
    """True for the parts of one module, or of a module and its parent: they
    meet at a joint (the link tip carries the child's bracket, the plate sits
    on a horn), so an overlap there is the interface, not a stalled print."""
    if a == b:
        return True
    return a >= 0 and b >= 0 and (morph.nodes[a].parent == b
                                  or morph.nodes[b].parent == a)


def _touches(a, b) -> bool:
    return not (a.min.X > b.max.X or b.min.X > a.max.X or
                a.min.Y > b.max.Y or b.min.Y > a.max.Y or
                a.min.Z > b.max.Z or b.min.Z > a.max.Z)


def report(morph, records=None, servo_step=None):
    """([clash], [horn gap]): (label, other, mm3) for the pairs that share more
    than MIN_OVERLAP, and (label, mm) for every link horn. A clean body gives
    one row per horn and no clash."""
    records = placements(morph) if records is None else records
    frames = spawn_frames(morph)
    body = servo_body(servo_step)
    items = []
    made = {}
    for p in records:
        pos, rot = frames[p.body]
        if p.name not in made:            # one solid per distinct STL
            made[p.name] = p.make()
        items.append((p.label, p.owner,
                      _moved(pos + rot @ np.array(p.pos), rot @ p.rot, made[p.name])))
    for i, node in morph.nodes.items():
        if node.kind in ("joint", "wheel", "segment"):
            pos, rot = frames[f"n{i}"]
            items.append((f"servo{i}", i, _moved(pos, rot, body)))
    boxes = [solid.bounding_box() for _, _, solid in items]
    clashes = []
    for a, (label_a, owner_a, solid_a) in enumerate(items):
        for b in range(a + 1, len(items)):
            label_b, owner_b, solid_b = items[b]
            if _joined(owner_a, owner_b, morph) or not _touches(boxes[a], boxes[b]):
                continue
            volume = (solid_a & solid_b).volume
            if volume > MIN_OVERLAP:
                clashes.append((label_a, label_b, volume))
    return clashes, _horn_gaps(records)


def _horn_gaps(records) -> list:
    """Largest distance from a link horn hole to the nearest hole of the servo
    horn it is bolted to. Both patterns are read in the module frame, which is
    where the servo axis sits, so the check is the hole circle and its phase,
    not the thickness of the plate that moves its own pattern 3 mm sideways."""
    holes = np.array([(x, y, 0.0) for x, y in parts.horn_hole_centres()])
    out = []
    for p in records:
        if not p.horn:
            continue
        pattern = (holes @ p.rot.T + np.array(p.pos))[:, :2]
        gap = max(np.min(np.linalg.norm(holes[:, :2] - h, axis=1)) for h in pattern)
        out.append((p.label, float(gap)))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True)
    p.add_argument("--type", required=True, dest="cell")
    p.add_argument("--servo-step", default=None,
                   help="STEP model of the STS3215, instead of a box")
    a = p.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from replay import load
    _, elite = load(a.archive, a.cell)
    clashes, horns = report(mo.decode(elite.morph), servo_step=a.servo_step)
    print(f"{'check':<6} {'part':<16} {'other':<16} {'value':>9}")
    for left, right, volume in clashes:
        print(f"{'clash':<6} {left:<16} {right:<16} {volume:7.1f} mm3")
    for label, gap in horns:
        print(f"{'horn':<6} {label:<16} {'':<16} {gap:7.2f} mm")
    flagged = len(clashes) + sum(gap > MAX_GAP for _, gap in horns)
    print(f"{a.cell}: {len(clashes)} clash(es), {len(horns)} link horn(s), "
          f"{flagged} flagged")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())

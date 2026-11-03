"""
check_assembly.py
-----------------
Check that one robot of an archive can be printed and screwed together: no two
printed parts, and no printed part and servo body, share more than a cubic
millimetre, except at a screwed lap between parts that are bolted together (a
cradle on its servo, an elbow tongue on its link, a wedge on its tab: 0.5 mm or
less along its thinnest direction), and every link hole lines up with the servo
horn holes it is screwed to.

    python tools/check_assembly.py --archive a.pkl --type 7servos-mid-trot
    python tools/check_assembly.py --archive a.pkl --type <type> --no-walk
    python tools/check_assembly.py --archive a.pkl --type <type> --servo-step sts3215.step

The printed parts are build123d solids at the frames render_assembled.py puts
the meshes in (tools/assembly.py). Servo bodies are boxes with their output
axis where parts.py draws the horn holes; with `--servo-step` a STEP model
replaces the box (origin at the body centre, x along the body, z along the
output axis). Every pair of parts is compared, also a module and its parent:
the parts meet at a joint only where the design laps them, so an overlap there
is a printing error like any other. By default the check repeats the poses of a
straight walk at 0.5, 1.5, 2.5 and 3.5 s, with the elite's own controller, so a
part that only clashes while the joint turns is caught too.
"""

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from assembly import placements, printed_frames, printed_rest  # noqa: E402

from build123d import Box, Location, Pos, gp_Trsf          # noqa: E402

from darwinbot import builder, parts                       # noqa: E402
from darwinbot import morphology as mo                     # noqa: E402

MIN_OVERLAP = 12.0   # mm^3 shared by two solids before it counts as a clash (the whole-mm rounding of the printed measures leaves slivers below it)
LAP = 0.5            # mm: thinnest direction of a screwed lap (tongue on plate)
MAX_GAP = 0.3        # mm between a link hole centre and the nearest horn hole
WALK_TIMES = (0.5, 1.5, 2.5, 3.5)   # s of the straight walk that is checked


def _model_frames(model, data) -> dict:
    """{body: (position mm, rotation)} of a model at its current state."""
    out = {}
    for k in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, k)
        if name:
            out[name] = (np.array(data.xpos[k]) * 1000.0,
                         np.array(data.xmat[k]).reshape(3, 3).copy())
    return out


def spawn_frames(morph) -> dict:
    """{body: (position mm, rotation)} of the model at the spawn pose."""
    model = mujoco.MjModel.from_xml_string(builder.build_mjcf(morph)[0])
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    return _model_frames(model, data)


def walk_frames(morph, ctrl, payload: float = 0.0) -> list:
    """Body frames at WALK_TIMES of a straight walk, or [] when the body is not
    buildable (the spawn check then has to explain it)."""
    from darwinbot.sim import Episode
    from darwinbot.terrain import get_terrains
    ep = Episode(morph, get_terrains("flat")[0], payload=payload)
    if not ep.buildable:
        return []
    cpg = ep.reset(ctrl, command=0.0)
    out, t0 = [], 0.0
    for t in WALK_TIMES:
        for _ in range(int(round((t - t0) / ep.model.opt.timestep))):
            ep.step(cpg)
        t0 = t
        out.append(_model_frames(ep.model, ep.data))
    return out


def _moved(pos, rot, solid):
    """`solid` from its own frame to `pos` (mm) and `rot` in the world."""
    trsf = gp_Trsf()
    trsf.SetValues(*rot[0], pos[0], *rot[1], pos[1], *rot[2], pos[2])
    return solid.moved(Location(trsf))


def servo_body(step_path=None, axis: str = "pitch"):
    """Servo body in the module frame of a joint about `axis`: the output axis
    is +Z at x = SERVO_AXIS_X behind the origin for a pitch joint, turned the
    same way as the cradle for yaw and roll."""
    if step_path:
        from build123d import import_step
        return parts._axis_rot(axis) * (
            Pos(-parts.SERVO_AXIS_X, 0, 0) * import_step(step_path))
    centre, half = parts.servo_body(axis)
    return Pos(*centre) * Box(*(2.0 * np.array(half)))


def _touches(a, b) -> bool:
    return not (a.min.X > b.max.X or b.min.X > a.max.X or
                a.min.Y > b.max.Y or b.min.Y > a.max.Y or
                a.min.Z > b.max.Z or b.min.Z > a.max.Z)


def _thinness(piece) -> float:
    """Thinnest side of the axis-aligned box around `piece`, in mm. The bounding
    box of a build123d shape is padded by the modelling tolerance (0.1 mm per
    side), which would turn a 0.45 mm tongue into 0.55, so the box of the
    vertices is used when the padded one is within 0.2 mm of it."""
    box = piece.bounding_box().size
    loose = min(box.X, box.Y, box.Z)
    pts = np.array([(v.X, v.Y, v.Z) for v in piece.vertices()])
    if len(pts) < 4:
        return loose
    tight = float(np.min(pts.max(axis=0) - pts.min(axis=0)))
    return tight if loose - tight <= 0.25 else loose


def _shared(solid_a, solid_b, frame_a=None, frame_b=None, lap=True) -> float:
    """mm^3 the two solids really share: a screwed lap passes, a clash does not.
    An empty or face-only intersection shares nothing; for a pair that is
    screwed together (`lap`), an overlap thinner than LAP along its thinnest
    direction is a tongue bolted on a plate or a servo seated in its cradle.
    The thickness is read in the frame of each part ((position, rotation)
    given as `frame_a` and `frame_b`), so a pose that tilts the pair does not
    thicken a lap; the world frame is used when no frame is given."""
    common = solid_a & solid_b
    if common is None:
        return 0.0
    total = 0.0
    for piece in common.solids():
        if piece.volume <= MIN_OVERLAP:
            continue
        if lap:
            thin = []
            for frame in (frame_a, frame_b):
                local = piece if frame is None else _moved(
                    -frame[1].T @ np.asarray(frame[0]), frame[1].T, piece)
                thin.append(_thinness(local))
            if min(thin) <= LAP:
                continue
        total += piece.volume
    return total


def lap_pairs(morph, records) -> set:
    """Labels of the pairs that are screwed together, and so may overlap by up
    to LAP: a cradle with its servo, the torso plate and its wedge, a wedge
    with the cradle foot, the part that prints a cradle (link tip or elbow
    plate) with the servo in it, an elbow plate lapped on its link tip, and a
    foot with the spigot of the link that holds it. A plain link rigid with
    another is printed in its owner, so its labels are the owner's."""
    from darwinbot.export import print_owner
    out = set()
    label = {p.label for p in records}

    def link(j):
        return f"link{print_owner(morph, j)}"

    for i, node in morph.nodes.items():
        servo = f"servo{i}"
        for name in (f"cradle{i}", f"wedge{i}"):
            if name in label:
                out.add(frozenset((name, servo)))
                out.add(frozenset((name, "torso_plate")))
        out.add(frozenset((f"cradle{i}", f"wedge{i}")))
        if node.kind == "foot" and node.parent > 0:
            out.add(frozenset((link(node.parent), f"foot{i}")))
        if node.kind in ("joint", "link"):
            out.add(frozenset((link(i), f"foot{i}")))
        if node.slot == 0 and node.parent > 0:
            out.add(frozenset((link(node.parent), servo)))
        if node.slot and node.parent > 0 and node.parent in morph.nodes \
                and morph.nodes[node.parent].kind in ("joint", "link", "spring"):
            elbow = f"elbow{node.parent}_{i}"
            out.add(frozenset((elbow, servo)))
            out.add(frozenset((elbow, link(node.parent))))
    return out


def _build(morph, records, step_path):
    """([(label, solid)], frames) with one solid per printed part and servo."""
    made = {}
    solids = []
    for p in records:
        if p.name not in made:                    # one solid per distinct STL
            made[p.name] = p.make()
        solids.append((p.label, p.body, p.pos, p.rot, made[p.name]))
    bodies = {}
    servo = []
    for i, node in morph.nodes.items():
        if node.kind in ("joint", "wheel", "segment"):
            # the servo body is fixed to the parent's link, at the joint frame:
            # the child body turns about it, the servo does not
            off, rot = printed_rest(morph, node)
            parent = "torso" if node.parent == 0 else f"n{node.parent}"
            axis = "yaw" if node.kind == "wheel" else node.axis
            if axis not in bodies:
                bodies[axis] = servo_body(step_path, axis)
            servo.append((f"servo{i}", parent, tuple(off), rot,
                          bodies[axis]))
    return solids + servo


def report(morph, records=None, servo_step=None, ctrl=None, walk=False,
           payload: float = 0.0, frames=None, poses_out=None):
    """([clash], [horn gap]): (label, other, mm3) for the pairs that share more
    than MIN_OVERLAP outside a screwed lap, and (label, mm) for every link horn.
    A clean body gives one row per horn and no clash. With `walk` the poses of
    a straight walk (`ctrl`, `payload`) are checked as well. A list given as
    `poses_out` receives, for each clash, the index of the first pose it shows
    in (0 is the spawn pose)."""
    records = placements(morph) if records is None else records
    posed = _build(morph, records, servo_step)
    laps = lap_pairs(morph, records)
    poses = [spawn_frames(morph)] if frames is None else [frames]
    if walk and frames is None:
        poses += walk_frames(morph, ctrl, payload)
    clashes = []
    seen = set()
    for k, pose in enumerate(poses):
        pose = printed_frames(morph, pose)
        frames = [(np.array(pose[body][0]) + pose[body][1] @ np.array(pos),
                   pose[body][1] @ rot) for _, body, pos, rot, _ in posed]
        items = [(label, _moved(*frame, solid))
                 for (label, _, _, _, solid), frame in zip(posed, frames)]
        boxes = [solid.bounding_box() for _, solid in items]
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                if not _touches(boxes[a], boxes[b]):
                    continue
                volume = _shared(items[a][1], items[b][1], frames[a], frames[b],
                                 frozenset((items[a][0], items[b][0])) in laps)
                if volume and (items[a][0], items[b][0]) not in seen:
                    seen.add((items[a][0], items[b][0]))
                    clashes.append((items[a][0], items[b][0], volume))
                    if poses_out is not None:
                        poses_out.append(k)
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
    p.add_argument("--no-walk", dest="walk", action="store_false",
                   help="check the spawn pose only")
    a = p.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from replay import load
    archive, elite = load(a.archive, a.cell)
    clashes, horns = report(mo.decode(elite.morph), servo_step=a.servo_step,
                           ctrl=elite.ctrl, walk=a.walk,
                           payload=getattr(archive, "payload", 0.0))
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

"""
builder.py
----------
Morphology tree + terrain -> MuJoCo MJCF string, plus the actuator layout the
controller needs to address each motor.

Frames: X forward, Y left, Z up. Every module is a body placed from its
parent's slot (offset, slot rotation, roll); joint axes are in the module
frame (pitch Z, yaw Y, roll X, negated when `flip` is set), so joint angle
zero is the straight rest pose. Wheel speed is positive when the robot rolls
forward.

The layout is a list of morphology.Actuator(slot, kind, axis) in model order
(depth first, children by slot), which is the order of the <actuator> block.

Parts are real: every joint is a Feetech STS3215 (12 V) with the torque and
speed limits of its data sheet, minus a safety margin. Each servo adds a 60 g
box, the torso carries a payload (ESP32, 2S LiPo, regulator), and links are
printed. The torso is the contour polygon split into eight triangular prisms
(centre, v_k, v_k+1), so MuJoCo only sees convex geoms. A child on a side slot
of a link hangs from a printed elbow plate that bridges the child's elbow
offset, so no module floats. See docs/hardware-dados.md.

Self-collision is ON: robot geoms (contype 2, conaffinity 3) touch each other
and the terrain (1, 1). MuJoCo already skips parent/child pairs. `payload` (kg) is extra mass on the torso's payload box, for carrying tasks.
A body whose
geoms already touch at spawn is not buildable (see sim.is_buildable).
"""

import re
from typing import Tuple

import numpy as np

from darwinbot import ablate, hardware
from darwinbot.controller import SERVO_SPEED, SERVO_TORQUE
from darwinbot.morphology import (AXIS_LOCAL, CASTER_FORK_SIZE, CASTER_WHEEL_T,
                                  CLAW_LENGTH, CLAW_RADIUS, CLAW_SPREAD,
                                  ELECTRONICS_BOX, ELBOW_BOSS_R, SEGMENT_HEIGHT,
                                  SERVO_BOX, STRUT_HALF_W, TAB_SIZE, TONGUE_HALF_H,
                                  TONGUE_T, PLATE_THICK, cradle_box, fork_boxes,
                                  servo_box,
                                  SKID_SIZE, SPIKE_LENGTH, SPIKE_RADIUS,
                                  SPRING_DAMPING, TAIL_BAR_SIZE, TAIL_TIP_RADIUS,
                                  TORSO_HEIGHT, FACE_SLOTS, SERVO_AXIS_OFFSET, Actuator, ActuatorLayout,
                                  Morphology, elbow_angle, elbow_distance, link_path, section_radius,
                                  link_tip, plate_thick, segment_box,
                                  spring_stiffness)

SHELL_DENSITY  = hardware.MATERIAL.shell_density   # kg/m^3, printed shell with infill
LIMB_DENSITY   = hardware.MATERIAL.limb_density
LINK_DENSITY   = hardware.MATERIAL.link_density    # kg/m^3, solid PLA printed link
PAYLOAD_MASS   = hardware.ELECTRONICS.mass         # kg: ESP32 + 3S LiPo + regulator
SERVO_MASS     = hardware.SERVO.mass + hardware.SERVO.bracket_mass   # kg: the servo plus its printed bracket
ROBOT_FRICTION = "1.0 0.02 0.001"
SPAWN_CLEARANCE = 0.02   # m of air under the lowest foot at spawn

JOINT_RANGE = {"yaw": 0.9, "pitch": 1.3, "roll": 0.9}   # +/- rad
# Back-EMF as viscous damping: torque falls linearly from SERVO_TORQUE at rest
# to zero at SERVO_SPEED, the datasheet no-load speed. With 0.56 the servo
# topped out at 2.3 rad/s, under half the real one, and any stride faster than
# that turned into saturated torque reversals.
JOINT_DAMPING  = SERVO_TORQUE / SERVO_SPEED
JOINT_FRICTION = hardware.SERVO.joint_friction
JOINT_ARMATURE = hardware.SERVO.joint_armature
JOINT_KP       = hardware.SERVO.joint_kp
SPRING_RANGE   = 1.2          # +/- rad, a TPU blade bends little
WHEEL_KV       = hardware.SERVO.wheel_kv          # N*m per rad/s, saturates at SERVO_TORQUE
WHEEL_DAMPING  = JOINT_DAMPING  # same motor: torque falls to zero at the no-load speed
CASTER_DAMPING = 0.01         # free caster wheel on a passive hinge

ROBOT_GEOM = f'contype="2" conaffinity="3" friction="{ROBOT_FRICTION}"'
# Priority 2 beats the terrain (priority 1), so these frictions win the
# contact. They scale with the terrain's friction so a skid is still slippery
# on ice and a spike still grips better than a plain foot.
SKID_FACTOR  = 0.15
SPIKE_FACTOR = 1.5
# Hard printed or moulded plastic (servo housings, PLA links, elbows, torso,
# tail, caster fork) on the floor: about 0.3 to 0.4, against 1.0 for the TPU
# feet and wheels. Without it a servo housing grips like a foot.
PLASTIC_FACTOR = hardware.MATERIAL.plastic_factor


def plastic_factor() -> float:
    return 1.0 if ablate.active("friction") else PLASTIC_FACTOR


def _special_geom(factor: float, terrain_friction: float) -> str:
    return (f'contype="2" conaffinity="3" priority="2" '
            f'friction="{factor * terrain_friction:.4f} 0.02 0.001"')
LINK_RGBA  = "0.55 0.60 0.70 1"
SPRING_RGBA = "0.95 0.50 0.10 1"   # orange: a passive flexure, no servo


def _f(*values) -> str:
    return " ".join(f"{v:.4f}" for v in values)


def spawn_height(morph: Morphology) -> float:
    """Torso centre height that puts the lowest point of the body just above z = 0."""
    return -morph.lowest_z() + SPAWN_CLEARANCE


# Collision groups: the terrain is contype 1 / conaffinity 1 and every printed
# part contype 2 / conaffinity 3, so parts touch the terrain and each other.
# The torso volume (a 60 mm prism that stands for the plate, the electronics
# and the room around them) is contype 4 / conaffinity 1: it meets the terrain
# only, as the printed robot has nothing solid above its plate for a limb to
# bump into; what limbs meet is the plate and its tabs, which are parts.
TORSO_VOLUME = 'contype="4" conaffinity="1" group="1"'


def _torso_volume(friction: float) -> str:
    return (f'{TORSO_VOLUME} priority="2" '
            f'friction="{plastic_factor() * friction:.4f} 0.02 0.001"')


PART_ONLY = 'density="0"'          # collision shape only: its mass is in the servo or link


def _torso(morph: Morphology, friction: float = 1.0) -> Tuple[str, str]:
    """Torso mesh assets and geoms: one triangular prism per contour edge,
    each with the mass of its own volume of shell. The prisms meet the terrain
    only; the printed plate (a sheet of the same contour below the joint
    plane) and a tab under every mount meet the limbs."""
    from darwinbot import export
    poly = morph.torso_polygon()
    h = TORSO_HEIGHT / 2
    drop = export.plate_drop(morph) / 1000.0
    top, bottom = -drop, -(drop + PLATE_THICK)
    areas = [abs(a[0] * b[1] - b[0] * a[1]) / 2
             for a, b in zip(poly, poly[1:] + poly[:1])]
    assets, geoms = [], []
    plastic = _special_geom(plastic_factor(), friction)
    for k, (a, b) in enumerate(zip(poly, poly[1:] + poly[:1])):
        verts = [(0, 0, -h), (a[0], a[1], -h), (b[0], b[1], -h),
                 (0, 0, h), (a[0], a[1], h), (b[0], b[1], h)]
        vertex = _f(*[v for p in verts for v in p])
        assets.append(f'<mesh name="torso{k}" vertex="{vertex}"/>')
        mass = SHELL_DENSITY * areas[k] * TORSO_HEIGHT
        geoms.append(f'<geom type="mesh" mesh="torso{k}" mass="{mass:.6f}" '
                     f'rgba="0.25 0.35 0.75 1" {_torso_volume(friction)}/>')
        sheet = [(0, 0, bottom), (a[0], a[1], bottom), (b[0], b[1], bottom),
                 (0, 0, top), (a[0], a[1], top), (b[0], b[1], top)]
        vertex = _f(*[v for p in sheet for v in p])
        assets.append(f'<mesh name="plate{k}" vertex="{vertex}"/>')
        geoms.append(f'<geom name="plate{k}" type="mesh" mesh="plate{k}" {PART_ONLY} '
                     f'rgba="0.25 0.35 0.75 1" {plastic}/>')
    core = morph.nodes[0]
    for slot, c in sorted(morph.child_map().get(0, {}).items()):
        if slot in FACE_SLOTS:
            continue                       # a face mount is drilled through the plate
        off, rot = morph.slot_frame(core, slot, morph.nodes[c])
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        centre = (off[0] - SERVO_AXIS_OFFSET * np.cos(yaw),
                  off[1] - SERVO_AXIS_OFFSET * np.sin(yaw), (top + bottom) / 2)
        geoms.append(f'<geom name="tab{c}" type="box" pos="{_f(*centre)}" '
                     f'size="{_f(TAB_SIZE[0] / 2, TAB_SIZE[1] / 2, PLATE_THICK / 2)}" '
                     f'xyaxes="{_f(np.cos(yaw), np.sin(yaw), 0, -np.sin(yaw), np.cos(yaw), 0)}" '
                     f'{PART_ONLY} rgba="0.25 0.35 0.75 1" {plastic}/>')
    return "\n    ".join(assets), "\n      ".join(geoms)


def _segment_geoms(node, friction: float = 1.0) -> str:
    """Second torso plate of a segment, centred after its spine servo."""
    centre, half = segment_box(node)
    mass = SHELL_DENSITY * node.length * node.width * SEGMENT_HEIGHT
    return (f'<geom name="segment{node.id}" type="box" pos="{_f(*centre)}" '
            f'size="{_f(*half)}" mass="{mass:.6f}" '
            f'rgba="0.25 0.35 0.75 1" {_special_geom(plastic_factor(), friction)}/>')


def _frame_attrs(morph: Morphology, node) -> str:
    off, rot = morph.rel_frame(node)
    return f'pos="{_f(*off)}" xyaxes="{_f(*rot[:, 0], *rot[:, 1])}"'


def _servo_box(i: int, axis: str, friction: float = 1.0) -> str:
    half, centre = servo_box(axis)
    return (f'<geom name="servo{i}" type="box" pos="{_f(*centre)}" size="{_f(*half)}" mass="{SERVO_MASS}" rgba="0.15 0.15 0.15 1" '
            f'{_special_geom(plastic_factor(), friction)}/>')


# Look only: no mass, no contact, render group 1. They show the printed
# bracket around the servo and the horn on its output face, which the physics
# approximates with the servo box and a link starting at the joint axis.
VISUAL_GEOM = 'contype="0" conaffinity="0" group="1" density="0"'
BRACKET_RGBA = "0.55 0.60 0.70 1"
HORN_RGBA = "0.80 0.80 0.82 1"
HORN_RADIUS, HORN_THICK = 0.011, 0.003


def _cradle_geom(i: int, axis: str, friction: float = 1.0) -> str:
    """The printed cradle around the servo, a collision shape without mass of
    its own (SERVO_MASS already counts the printed holder). It lives next to
    the servo box, in the part the servo is bolted to."""
    half, centre = cradle_box(axis)
    return (f'<geom name="cradle{i}" type="box" pos="{_f(*centre)}" size="{_f(*half)}" '
            f'{PART_ONLY} rgba="{BRACKET_RGBA}" {_special_geom(plastic_factor(), friction)}/>')


def _servo_dress(axis: str) -> str:
    """The horns on the two faces the joint axis points out of (the link
    clasps both). Look only."""
    half, centre = servo_box(axis)
    a = np.abs(np.asarray(AXIS_LOCAL[axis], dtype=float))
    face = float(a @ half)
    return "\n".join(
        f'<geom type="cylinder" fromto="{_f(*(s * a * face), *(s * a * (face + HORN_THICK)))}" '
        f'size="{HORN_RADIUS}" rgba="{HORN_RGBA}" {VISUAL_GEOM}/>' for s in (1, -1))


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _xyaxes(rot: np.ndarray) -> str:
    return f'xyaxes="{_f(*rot[:, 0], *rot[:, 1])}"'


def _boss(name: str, node, point, rgba: str, contact: str) -> str:
    """Round end boss of a printed link plate (parts._bar), at a point of the
    link path."""
    r = max(node.width / 2, ELBOW_BOSS_R)
    t = plate_thick(node) / 2
    return (f'<geom name="{name}" type="cylinder" '
            f'fromto="{_f(point[0], point[1], -t, point[0], point[1], t)}" '
            f'size="{r:.4f}" {PART_ONLY} rgba="{rgba}" {contact}/>')


def _box_geom(name: str, centre, half, contact: str, rgba: str = LINK_RGBA,
              rot: np.ndarray = None) -> str:
    axes = "" if rot is None else f' {_xyaxes(rot)}'
    return (f'<geom name="{name}" type="box" pos="{_f(*centre)}"{axes} '
            f'size="{_f(*half)}" {PART_ONLY} rgba="{rgba}" {contact}/>')


def _link_geoms(node, rgba: str = LINK_RGBA, contact: str = ROBOT_GEOM,
                tip_cap: bool = True) -> str:
    """One box geom per piece of the link path; the last piece carries the
    tip name that sim.terminal_geoms uses for a free end. contact: the PLA
    friction for a link, ROBOT_GEOM for a TPU spring. The printed plate also
    has a round boss at every end of a piece (`tip_cap` False leaves out the
    one at the tip, where a servo cradle or a foot takes its place), and a
    joint's link starts as a U fork that clasps its servo, drawn as the arms
    and the bridge of parts._u_root."""
    pts = link_path(node)
    out = []
    for k, (a, b) in enumerate(zip(pts, pts[1:])):
        d = b - a
        rot = _rot_z(float(np.arctan2(d[1], d[0])))
        name = f"tip{node.id}" if k == len(pts) - 2 else f"lk{node.id}_{k}"
        out.append(f'<geom name="{name}" type="box" pos="{_f(*((a + b) / 2))}" '
                   f'{_xyaxes(rot)} '
                   f'size="{_f(np.linalg.norm(d) / 2, node.width / 2, plate_thick(node) / 2)}" '
                   f'density="{LINK_DENSITY}" rgba="{rgba}" {contact}/>')
    for k, p in enumerate(pts):
        if (k == 0 and node.kind == "joint") or (k == len(pts) - 1 and not tip_cap):
            continue
        out.append(_boss(f"boss{node.id}_{k}", node, p, rgba, contact))
    if node.kind == "joint":
        for k, (centre, half) in enumerate(fork_boxes(node.axis, node.width, node.thick)):
            out.append(_box_geom(f"fork{node.id}_{k}", centre, half, contact, rgba))
    return "\n        ".join(out)


def _strut_height(morph: Morphology, child) -> float:
    """Half height, across the link plane, of the strut of an elbow plate that
    carries a servo: the cradle's extent on both sides of the plane less half a
    millimetre (parts.elbow_plate)."""
    axis = "yaw" if child.kind == "wheel" else child.axis
    half, centre = cradle_box(axis)
    off, rot = morph.rel_frame(child)
    mid = float(off[2] + rot[2] @ centre)
    ext = float(np.abs(rot[2]) @ half)
    return max(0.001, min(-(mid - ext), mid + ext) - 0.0005)


def _elbow_geom(morph: Morphology, parent, child, slot: int,
                friction: float = 1.0) -> str:
    """Printed elbow from the link tip to the side slot `slot` (1 left, 2
    right), where the child module is mounted. It follows the tip rotation of
    a bent link. A servo child hangs on a strut with a tongue bolted to the fin
    of the link tip, as parts.elbow_plate draws it (its cradle is the child's
    own geom); anything else on a bar with round bosses."""
    side = 1.0 if slot == 1 else -1.0
    d = elbow_distance(child)
    tip, rot = link_tip(parent)
    plastic = _special_geom(plastic_factor(), friction)
    near, w = -ELBOW_BOSS_R, max(parent.width, 2 * ELBOW_BOSS_R)
    if child.kind in ("joint", "segment", "wheel"):
        from darwinbot import export
        axis = "yaw" if child.kind == "wheel" else child.axis
        at = export.elbow_trim(export._mm(d), int(round(np.degrees(elbow_angle(child)))),
                               axis, export.servo_roll(morph, child.id, int(side)))[0] / 1000.0
        hz = _strut_height(morph, child)
        # the strut stops at the back wall of the cradle it carries
        end = max(at + 0.002, d - 0.0381)
        centre = tip + rot @ np.array([0.0, side * (at + end) / 2, 0.0])
        # the strut keeps the mass of the plate it replaces
        mass = LINK_DENSITY * w * (d - near) * plate_thick(parent)
        strut = _box_geom(f"strut{child.id}", centre,
                          (STRUT_HALF_W, max(1e-4, (end - at) / 2), hz), plastic, rot=rot)
        strut = strut.replace(PART_ONLY, f'mass="{mass:.6f}"')
        tongue_c = tip + rot @ np.array([0.0, side * (at - TONGUE_T / 2), 0.0])
        tongue = _box_geom(f"tongue{child.id}", tongue_c,
                           (0.0153, TONGUE_T / 2, TONGUE_HALF_H), plastic, rot=rot)
        return f"{strut}\n        {tongue}"
    # the printed plate starts with a round boss around the link tip
    pos = tip + rot @ np.array([0.0, side * (near + d) / 2, 0.0])
    return (f'<geom name="elbow{child.id}" type="box" '
            f'pos="{_f(*pos)}" {_xyaxes(rot)} '
            f'size="{_f(w / 2, (d - near) / 2, plate_thick(parent) / 2)}" '
            f'density="{LINK_DENSITY}" rgba="{LINK_RGBA}" {plastic}/>')


def _foot_shape_geoms(node, friction: float = 1.0) -> str:
    """Foot of the shape picked by the AXIS gene. All TPU. The pad is a disc
    with its axis along local X (the limb direction), so its flat face looks
    down when the limb points down; the claws spread in the local XY plane."""
    r = node.foot_rad
    geom = f'density="{LIMB_DENSITY}" rgba="0.2 0.2 0.2 1"'
    if node.foot_shape == "blade":
        return (f'<geom type="box" size="{_f(r, r / 3 / 2, r / 2)}" '
                f'{geom} {ROBOT_GEOM}/>')
    if node.foot_shape == "pad":
        return (f'<geom type="cylinder" fromto="{_f(-r / 6, 0, 0, r / 6, 0, 0)}" '
                f'size="{r:.4f}" {geom} {ROBOT_GEOM}/>')
    if node.foot_shape == "claw":
        out = []
        for a in (-CLAW_SPREAD, 0.0, CLAW_SPREAD):
            end = np.array([np.cos(a), np.sin(a), 0.0]) * CLAW_LENGTH * r
            out.append(f'<geom type="capsule" fromto="{_f(0, 0, 0, *end)}" '
                       f'size="{CLAW_RADIUS * r:.4f}" {geom} '
                       f'{_special_geom(SPIKE_FACTOR, friction)}/>')
        return "\n        ".join(out)
    return (f'<geom type="sphere" size="{r:.4f}" density="{LIMB_DENSITY}" '
            f'rgba="0.2 0.2 0.2 1" {ROBOT_GEOM}/>')


def _foot_hub(i: int, back: float) -> str:
    """The printed foot's socket hub (parts._socketed): a 16 mm cylinder 12 mm
    long behind the shape, over the end of the limb. Collision only."""
    return (f'<geom name="hub{i}" type="cylinder" fromto="{_f(back, 0, 0, back - 0.012, 0, 0)}" '
            f'size="0.008" {PART_ONLY} rgba="0.2 0.2 0.2 1" {ROBOT_GEOM}/>')


def _foot_geoms(node, friction: float = 1.0) -> str:
    """Foot shape plus the stem or hub the printed cap has behind it."""
    r = node.foot_rad
    out = _foot_shape_geoms(node, friction)
    if node.foot_shape == "sphere":
        rs = max(r, 0.0086)
        out += (f'\n        <geom name="hub{node.id}" type="cylinder" fromto="{_f(-10 * rs / 9, 0, 0, 0, 0, 0)}" '
                f'size="{rs:.4f}" {PART_ONLY} rgba="0.2 0.2 0.2 1" {ROBOT_GEOM}/>')
    else:
        back = {"blade": -r, "pad": -r / 6, "claw": 0.0}[node.foot_shape]
        out += "\n        " + _foot_hub(node.id, back)
    return out


def _cap_geoms(node) -> str:
    """The TPU cap printed over the free end of a link (a sphere foot of the
    section radius), at the tip of its path."""
    tip, rot = link_tip(node)
    rs = max(float(section_radius(node)), 0.0086)
    back = tip + rot @ np.array([-10 * rs / 9, 0.0, 0.0])
    return (f'<geom name="cap{node.id}" type="sphere" pos="{_f(*tip)}" size="{rs:.4f}" '
            f'{PART_ONLY} rgba="0.2 0.2 0.2 1" {ROBOT_GEOM}/>\n        '
            f'<geom name="hub{node.id}" type="cylinder" fromto="{_f(*back, *tip)}" size="{rs:.4f}" '
            f'{PART_ONLY} rgba="0.2 0.2 0.2 1" {ROBOT_GEOM}/>')


def _node_xml(morph: Morphology, node, frames, children, actuators: list,
              layout: ActuatorLayout, friction: float = 1.0, ranges=None) -> str:
    """MJCF body of one module and, recursively, of everything below it."""
    i = node.id
    ranges = {} if ranges is None else ranges
    inner = ""
    plastic = _special_geom(plastic_factor(), friction)
    tip_child = children.get(i, {}).get(0)
    # a servo cradle or a foot takes the place of the boss at the link tip
    tip_cap = tip_child is not None and morph.nodes[tip_child].kind not in (
        "joint", "segment", "wheel", "foot")
    if node.kind == "joint":
        name = f"j{i}"
        rng = ranges.get(i, (-JOINT_RANGE[node.axis], JOINT_RANGE[node.axis]))
        sign = -1.0 if node.flip else 1.0
        axis = sign * np.array(AXIS_LOCAL[node.axis])
        actuators.append(
            f'<position name="{name}" joint="{name}" kp="{JOINT_KP}" kv="0" '
            f'ctrlrange="{rng[0]:.4f} {rng[1]:.4f}" forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        layout.append(Actuator(i, "joint", node.axis))
        inner = (f'<joint name="{name}" type="hinge" axis="{_f(*axis)}" '
                 f'range="{rng[0]:.4f} {rng[1]:.4f}"/>\n{_link_geoms(node, contact=plastic, tip_cap=tip_cap)}')
    elif node.kind == "segment":
        name = f"j{i}"
        rng = ranges.get(i, (-JOINT_RANGE[node.axis], JOINT_RANGE[node.axis]))
        sign = -1.0 if node.flip else 1.0
        axis = sign * np.array(AXIS_LOCAL[node.axis])
        actuators.append(
            f'<position name="{name}" joint="{name}" kp="{JOINT_KP}" kv="0" '
            f'ctrlrange="{rng[0]:.4f} {rng[1]:.4f}" forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        layout.append(Actuator(i, "joint", node.axis))
        inner = (f'<joint name="{name}" type="hinge" axis="{_f(*axis)}" '
                 f'range="{rng[0]:.4f} {rng[1]:.4f}"/>\n{_segment_geoms(node, friction)}')
    elif node.kind == "link":
        inner = _link_geoms(node, contact=plastic, tip_cap=tip_cap)
    elif node.kind == "spring":
        sign = -1.0 if node.flip else 1.0
        axis = sign * np.array(AXIS_LOCAL[node.axis])
        srng = ranges.get(i, (-SPRING_RANGE, SPRING_RANGE))
        inner = (f'<joint name="s{i}" type="hinge" axis="{_f(*axis)}" '
                 f'stiffness="{spring_stiffness(node):.8g}" '
                 f'damping="{SPRING_DAMPING}" frictionloss="0" armature="0" '
                 f'range="{srng[0]:.4f} {srng[1]:.4f}"/>\n'
                 f'{_link_geoms(node, SPRING_RGBA, tip_cap=tip_cap)}')
    elif node.kind == "wheel":
        pos, rot = frames[i]
        name = f"w{i}"
        hub = morph.wheel_hub(node, pos, rot)
        half_t = node.wheel_width / 2
        axle = np.array([0.0, morph.wheel_axis_sign(rot), 0.0])
        actuators.append(
            f'<velocity name="{name}" joint="{name}" kv="{WHEEL_KV}" '
            f'ctrlrange="{-SERVO_SPEED} {SERVO_SPEED}" '
            f'forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        layout.append(Actuator(i, "wheel", "wheel"))
        inner = f"""{_servo_box(i, "yaw", friction)}
          {_cradle_geom(i, "yaw", friction)}
          <body name="{name}" pos="{_f(*hub)}">
            <joint name="{name}" type="hinge" axis="{_f(*axle)}" damping="{WHEEL_DAMPING}"/>
            <geom type="cylinder" fromto="{_f(0, -half_t, 0, 0, half_t, 0)}"
                  size="{node.wheel_rad:.4f}" density="{LIMB_DENSITY}"
                  rgba="0.1 0.1 0.1 1" {ROBOT_GEOM}/>
          </body>"""
    elif node.kind == "caster":
        pos, rot = frames[i]
        name = f"c{i}"
        hub = morph.caster_hub(node, pos, rot)
        half_t = CASTER_WHEEL_T / 2
        axle = np.array([0.0, morph.wheel_axis_sign(rot), 0.0])
        inner = f"""<geom name="fork{i}" type="box" size="{_f(*(np.array(CASTER_FORK_SIZE) / 2))}"
              density="{LINK_DENSITY}" rgba="{LINK_RGBA}" {plastic}/>
          <body name="{name}" pos="{_f(*hub)}">
            <joint name="{name}" type="hinge" axis="{_f(*axle)}" damping="{CASTER_DAMPING}"
                   frictionloss="0" armature="0"/>
            <geom type="cylinder" fromto="{_f(0, -half_t, 0, 0, half_t, 0)}"
                  size="{node.wheel_rad:.4f}" density="{LIMB_DENSITY}"
                  rgba="0.1 0.1 0.1 1" {ROBOT_GEOM}/>
          </body>"""
    elif node.kind == "foot":
        inner = _foot_geoms(node, friction)
    elif node.kind == "tail":
        mass = max(node.tail_mass, 1e-6)   # a zero-mass geom would be degenerate
        inner = (f'<geom name="tail{i}" type="box" pos="{_f(node.length / 2, 0, 0)}" '
                 f'size="{_f(node.length / 2, TAIL_BAR_SIZE[0] / 2, TAIL_BAR_SIZE[1] / 2)}" '
                 f'density="{LINK_DENSITY}" rgba="{LINK_RGBA}" {plastic}/>\n'
                 f'        <geom name="tailtip{i}" type="sphere" '
                 f'pos="{_f(node.length, 0, 0)}" size="{TAIL_TIP_RADIUS:.4f}" '
                 f'mass="{mass:.6f}" rgba="0.15 0.15 0.15 1" {plastic}/>')
    elif node.kind == "skid":
        inner = (f'<geom name="skid{i}" type="box" '
                 f'size="{_f(*(np.array(SKID_SIZE) / 2))}" '
                 f'density="{LINK_DENSITY}" rgba="0.2 0.25 0.30 1" {_special_geom(SKID_FACTOR, friction)}/>')
    elif node.kind == "spike":
        inner = (f'<geom name="spike{i}" type="capsule" '
                 f'fromto="{_f(0, 0, 0, SPIKE_LENGTH, 0, 0)}" '
                 f'size="{SPIKE_RADIUS:.4f}" density="{LINK_DENSITY}" '
                 f'rgba="{SPRING_RGBA}" {_special_geom(SPIKE_FACTOR, friction)}/>')
    if node.kind in ("joint", "link", "spring"):
        inner += "".join(f"\n{_elbow_geom(morph, node, morph.nodes[c], slot, friction)}"
                         for slot, c in sorted(children.get(i, {}).items()) if slot)
    if node.kind in ("joint", "link") and i not in children:
        inner += f"\n{_cap_geoms(node)}"
    # Actuators follow the depth first order of Morphology.order.
    below = "".join(
        _node_xml(morph, morph.nodes[c], frames, children, actuators, layout, friction,
                  ranges)
        for _, c in sorted(children.get(i, {}).items()))
    # A joint's servo body is fixed to the part it is mounted on, as on the
    # robot, where only the horns turn with the link: it lives in the parent
    # body, at the joint frame, so it never swings into its own mount.
    mount = ""
    if node.kind in ("joint", "segment"):
        mount = (f'\n      <frame {_frame_attrs(morph, node)}>\n        '
                 f'{_servo_box(i, node.axis, friction)}\n        {_cradle_geom(i, node.axis, friction)}\n        {_servo_dress(node.axis)}\n      </frame>')
    return f"""{mount}
      <body name="n{i}" {_frame_attrs(morph, node)}>
        {inner}{below}
      </body>"""


def build_mjcf(morph: Morphology, terrain_assets: str = "",
               terrain_world: str = "", spawn_x: float = 0.0,
               ground_z: float = 0.0,
               payload: float = 0.0,
               terrain_friction: float = 1.0,
               ranges=None) -> Tuple[str, ActuatorLayout]:
    """ranges: {node id: (lo, hi)} joint limits in rad; None measures them
    with joint_ranges, {} keeps the JOINT_RANGE defaults."""
    if ranges is None:
        ranges = {} if ablate.active("ranges") else joint_ranges(morph)
    friction = terrain_friction
    actuators: list = []
    layout: ActuatorLayout = []
    frames = morph.frames()
    children = morph.child_map()
    legs = "".join(
        _node_xml(morph, morph.nodes[c], frames, children, actuators, layout, friction,
                  ranges)
        for _, c in sorted(children.get(0, {}).items())
    )
    assert layout == morph.layout()
    z0 = ground_z + spawn_height(morph)
    actuator_xml = "\n    ".join(actuators)
    torso_assets, torso_geoms = _torso(morph, terrain_friction)

    xml = f"""<mujoco model="darwinbot">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast"/>
  <default>
    <geom condim="3"/>
    <joint damping="{JOINT_DAMPING}" frictionloss="{JOINT_FRICTION}" armature="{JOINT_ARMATURE}"/>
  </default>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.85 0.85 0.85" rgb2="0.70 0.70 0.70"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
    {terrain_assets}
    {torso_assets}
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {terrain_world}
    <body name="torso" pos="{_f(spawn_x, 0, z0)}">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="0 -1.8 0.9" xyaxes="1 0 0 0 0.45 0.9"/>
      {torso_geoms}
      <geom type="box" size="{_f(ELECTRONICS_BOX[0] / 2, ELECTRONICS_BOX[1] / 2, 0.015)}"
            mass="{PAYLOAD_MASS + payload}" contype="0" conaffinity="0" rgba="0.2 0.6 0.2 0.3"/>
      {legs}
    </body>
  </worldbody>
  <actuator>
    {actuator_xml}
  </actuator>
</mujoco>"""
    return xml, layout


# --------------------------------------------------------------------------
# Does it assemble? The printed parts are collision shapes, and these rules
# say which of them may meet: a servo in its cradle, a link in the U fork that
# clasps it, an elbow on the tip it is bolted to. Everything else that
# overlaps is a clash, in the spawn pose (sim.is_buildable) and in motion
# (joint_ranges, one joint at a time). MuJoCo never collides bodies that are parent and child, or
# welded together (a passive link, a foot, a tab: nothing turns between
# them), so those pairs are measured here with mj_geomDistance.
# --------------------------------------------------------------------------

HUB_RADIUS = 0.026          # m: a link overlaps the plate it is bolted to near its end
CLASH_DEPTH = 2e-4          # m: shared depth that counts as a clash
_MOUNT_ROLES = ("servo", "cradle", "strut", "tongue", "elbow")
_NAMED = re.compile(r"^(servo|cradle|strut|tongue|elbow|tab|tip|segment)(\d+)$")
_NAMED_K = re.compile(r"^(lk|boss|fork)(\d+)_(\d+)$")


class AssemblyRules:
    """Roles of the robot geoms of one model and which pairs may overlap."""

    def __init__(self, morph: Morphology, model):
        import mujoco
        self.morph = morph
        self.robot = model.geom_contype == 2
        ng = model.ngeom
        self.role = [""] * ng
        self.node = [-1] * ng
        self.part = [None] * ng
        body_node = {}
        self.body_of = {}
        for b in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name == "torso":
                body_node[b] = 0
            elif name[:1] in "nwc" and name[1:].isdigit():
                body_node[b] = int(name[1:])
                if name[0] == "n":
                    self.body_of[int(name[1:])] = b
        self.body_of[0] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        for g in range(ng):
            if not self.robot[g]:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            m, k = _NAMED.match(name), _NAMED_K.match(name)
            if m and m.group(1) in _MOUNT_ROLES:
                role, node = m.group(1), int(m.group(2))
                part = ("mount", node)
            elif m and m.group(1) in ("tip", "segment"):
                role, node, part = "link", int(m.group(2)), ("link", int(m.group(2)))
            elif m:                                        # tab
                role, node, part = "tab", int(m.group(2)), ("torso",)
            elif k and k.group(1) == "fork":
                role, node = "fork", int(k.group(2))
                part = ("link", node)
            elif k:
                role, node = "link", int(k.group(2))
                part = ("link", node)
            elif name.startswith("plate"):
                role, node, part = "plate", 0, ("torso",)
            else:
                node = body_node.get(int(model.geom_bodyid[g]), 0)
                role, part = "other", ("other", node)
                if morph.nodes[node].kind in ("caster", "tail", "skid", "spike"):
                    role = "free"       # no printed part is drawn for these yet
            self.role[g], self.node[g], self.part[g] = role, node, part
        self.parent = {i: n.parent for i, n in morph.nodes.items()}
        for g in range(ng):                # a passive bar is printed with the link it hangs on
            if self.role[g] == "elbow":
                self.part[g] = ("link", self.parent[self.node[g]])
        self.tip = {i for i, n in morph.nodes.items()
                    if n.slot == 0 and n.parent > 0
                    and morph.nodes[n.parent].kind in ("joint", "link", "spring")}
        self.on_plate = {i for i, n in morph.nodes.items()
                         if n.parent >= 0 and morph.nodes[n.parent].kind == "segment"}
        # pairs MuJoCo skips because nothing turns between the two bodies
        weld = model.body_weldid
        idx = np.flatnonzero(self.robot)
        self.fixed_pairs = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]
                            if weld[model.geom_bodyid[a]] == weld[model.geom_bodyid[b]]
                            and self.part[a] != self.part[b]]
        self._near = None

    def allowed(self, a: int, b: int, pos, xpos) -> bool:
        """True when geoms a and b may overlap at `pos` (world); `xpos` maps a
        node id to the world position of its frame."""
        if self.part[a] == self.part[b] or "free" in (self.role[a], self.role[b]):
            return True
        for (g, h) in ((a, b), (b, a)):
            rg, ng_, rh, nh = self.role[g], self.node[g], self.role[h], self.node[h]
            if rg in _MOUNT_ROLES:
                if nh == ng_ and rh in ("link", "other"):
                    return True                        # a servo in its cradle, under its link root
                p = self.parent.get(ng_)
                if rh == "link" and nh == p:
                    # the cradle of a servo at a link tip is printed with the link,
                    # an elbow plate is bolted to it, a spine plate carries its mounts
                    if (rg in ("strut", "tongue", "elbow") or ng_ in self.tip
                            or ng_ in self.on_plate):
                        return True
            if rg == "tab" and rh in ("link", "other") and nh == ng_:
                return True
            if rg == "other" and rh == "link" and nh == ng_:
                return True                            # a cap over its own link end
            if rg in ("link", "other") and rh == "link" and nh == self.parent.get(ng_):
                if np.linalg.norm(np.asarray(pos) - xpos[ng_]) <= (
                        HUB_RADIUS if rg == "link" else 0.04):
                    return True                        # plates lapped at the joint
        return False

    def clashes(self, model, data, fixed: bool = False, first: bool = False) -> list:
        """[(geom a, geom b, depth)] of the robot geoms that overlap where they
        must not, in the pose of `data` (forward kinematics done, and the
        parent filter of `model` off). `fixed` also measures the pairs that
        cannot move against each other, which only the spawn pose needs."""
        import mujoco
        out = []
        xpos = None
        n = data.ncon
        if n:
            g = data.contact.geom[:n]
            dist = data.contact.dist[:n]
            hit = np.flatnonzero(self.robot[g[:, 0]] & self.robot[g[:, 1]]
                                 & (dist < -CLASH_DEPTH))
            for c in hit:
                a, b = int(g[c, 0]), int(g[c, 1])
                if xpos is None:
                    xpos = self._xpos(data)
                if not self.allowed(a, b, data.contact.pos[c], xpos):
                    out.append((a, b, float(-dist[c])))
                    if first:
                        return out
        if fixed:
            if self._near is None:
                self._near = self._near_pairs(model, data)
            fromto = np.zeros(6)
            for a, b in self._near:
                d = mujoco.mj_geomDistance(model, data, a, b, 0.0, fromto)
                if d < -CLASH_DEPTH:
                    if xpos is None:
                        xpos = self._xpos(data)
                    if not self.allowed(a, b, (fromto[:3] + fromto[3:]) / 2, xpos):
                        out.append((a, b, float(-d)))
                        if first:
                            return out
        return out

    def _xpos(self, data) -> dict:
        return {i: data.xpos[b] for i, b in self.body_of.items()}

    def _near_pairs(self, model, data) -> list:
        """The fixed pairs whose bounding spheres meet in this pose (the
        pairs never move against each other, so the list holds for any pose)."""
        c, r = data.geom_xpos, model.geom_rbound
        return [(a, b) for a, b in self.fixed_pairs
                if np.linalg.norm(c[a] - c[b]) < r[a] + r[b] + 1e-3]

    def bodies(self, model, a: int, b: int) -> tuple:
        return int(model.geom_bodyid[a]), int(model.geom_bodyid[b])


SWEEP_STEP = np.radians(3.0)
SWEEP_MARGIN = np.radians(8.0)   # kept clear of the first clashing angle
_RANGES: dict = {}


def joint_ranges(morph: Morphology) -> dict:
    """{node id: (lo, hi)} rad for every servo joint and spring, inside their
    defaults. A printed robot is not allowed to turn into itself: each joint is
    swept on its own from 0 towards both ends and stops one step before the
    first pose where printed parts clash (the parent filter of MuJoCo is off
    here, so a link that cuts into its parent counts). Clashes that need
    several joints at once are not looked for: the parts of different limbs
    collide in the episode (self-collision is on), like a real robot hitting
    itself."""
    import mujoco
    from darwinbot.morphology import encode
    key = encode(morph).tobytes()
    if key in _RANGES:
        return _RANGES[key]
    xml, _ = build_mjcf(morph, ranges={})
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    data = mujoco.MjData(model)
    rules = AssemblyRules(morph, model)
    out: dict = {}
    nodes = [i for i, n in morph.nodes.items()
             if n.kind in ("joint", "segment", "spring")]
    names = {i: (f"s{i}" if morph.nodes[i].kind == "spring" else f"j{i}") for i in nodes}
    jid = {i: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[i]) for i in nodes}
    limit = {i: (SPRING_RANGE if morph.nodes[i].kind == "spring"
                 else JOINT_RANGE[morph.nodes[i].axis]) for i in nodes}

    def pose(q: dict):
        data.qpos[:] = model.qpos0
        for i, v in q.items():
            data.qpos[model.jnt_qposadr[jid[i]]] = v
        mujoco.mj_forward(model, data)

    pose({})
    if rules.clashes(model, data, fixed=True, first=True):
        _RANGES[key] = out      # not buildable: sim.is_buildable says so
        return out
    for i in nodes:
        ends = []
        for sign in (1.0, -1.0):
            reach, blocked = 0.0, None
            for a in np.arange(SWEEP_STEP, limit[i] + 1e-9, SWEEP_STEP):
                pose({i: sign * a})
                if rules.clashes(model, data, first=True):
                    blocked = a
                    break
                reach = a
            if blocked is not None:           # where between the two steps it starts
                lo, hi = reach, blocked
                for _ in range(5):
                    mid = (lo + hi) / 2
                    pose({i: sign * mid})
                    if rules.clashes(model, data, first=True):
                        hi = mid
                    else:
                        lo = mid
                reach = max(0.0, lo - SWEEP_MARGIN)
            ends.append(sign * reach)
        out[i] = (float(ends[1]), float(ends[0]))
    out = {i: (round(lo, 4), round(hi, 4)) for i, (lo, hi) in out.items()}
    if len(_RANGES) > 4096:
        _RANGES.clear()
    _RANGES[key] = out
    return out


def spawn_clashes(morph: Morphology) -> list:
    """[(node a, node b)] of the modules whose printed parts clash in the
    spawn pose, worst first (the same test as sim.is_buildable)."""
    import mujoco
    xml, _ = build_mjcf(morph, ranges={})
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rules = AssemblyRules(morph, model)
    hits = sorted(rules.clashes(model, data, fixed=True), key=lambda h: -h[2])
    return [(rules.node[a], rules.node[b]) for a, b, _ in hits]

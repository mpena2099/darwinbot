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

Parts are real: every joint is a Feetech STS3215 (7.4 V) with the torque and
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

from typing import Tuple

import numpy as np

from darwinbot.controller import SERVO_SPEED, SERVO_TORQUE
from darwinbot.morphology import (AXIS_LOCAL, CASTER_FORK_SIZE, CASTER_WHEEL_T,
                                  ELECTRONICS_BOX, SERVO_BOX, SKID_SIZE,
                                  SPIKE_LENGTH, SPIKE_RADIUS, SPRING_DAMPING,
                                  TAIL_BAR_SIZE, TAIL_TIP_RADIUS, TORSO_HEIGHT,
                                  Actuator, ActuatorLayout, Morphology,
                                  elbow_distance, plate_thick, spring_stiffness)

SHELL_DENSITY  = 300.0   # kg/m^3, printed shell with infill
LIMB_DENSITY   = 300.0
LINK_DENSITY   = 1240.0  # kg/m^3, solid PLA printed link
PAYLOAD_MASS   = 0.13    # kg: ESP32 + 2S LiPo + regulator
SERVO_MASS     = 0.060   # kg
ROBOT_FRICTION = "1.0 0.02 0.001"
SPAWN_CLEARANCE = 0.02   # m of air under the lowest foot at spawn

JOINT_RANGE = {"yaw": 0.9, "pitch": 1.3, "roll": 0.9}   # +/- rad
JOINT_DAMPING  = 0.56
JOINT_FRICTION = 0.068
JOINT_ARMATURE = 0.027
JOINT_KP       = 13.37
SPRING_RANGE   = 1.2          # +/- rad, a TPU blade bends little
WHEEL_KV       = 2.0          # N*m per rad/s, saturates at SERVO_TORQUE
WHEEL_DAMPING  = 0.05         # wheel spins freely, unlike a position servo
CASTER_DAMPING = 0.01         # free caster wheel on a passive hinge

ROBOT_GEOM = f'contype="2" conaffinity="3" friction="{ROBOT_FRICTION}"'
SERVO_GEOM = f'mass="{SERVO_MASS}" rgba="0.15 0.15 0.15 1" {ROBOT_GEOM}'
# Priority 2 beats the terrain (priority 1), so these frictions win the
# contact. They scale with the terrain's friction so a skid is still slippery
# on ice and a spike still grips better than a plain foot.
SKID_FACTOR  = 0.15
SPIKE_FACTOR = 1.5


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


def _torso(morph: Morphology) -> Tuple[str, str]:
    """Torso mesh assets and geoms: one triangular prism per contour edge,
    each with the mass of its own volume of shell."""
    poly = morph.torso_polygon()
    h = TORSO_HEIGHT / 2
    areas = [abs(a[0] * b[1] - b[0] * a[1]) / 2
             for a, b in zip(poly, poly[1:] + poly[:1])]
    assets, geoms = [], []
    for k, (a, b) in enumerate(zip(poly, poly[1:] + poly[:1])):
        verts = [(0, 0, -h), (a[0], a[1], -h), (b[0], b[1], -h),
                 (0, 0, h), (a[0], a[1], h), (b[0], b[1], h)]
        vertex = _f(*[v for p in verts for v in p])
        assets.append(f'<mesh name="torso{k}" vertex="{vertex}"/>')
        mass = SHELL_DENSITY * areas[k] * TORSO_HEIGHT
        geoms.append(f'<geom type="mesh" mesh="torso{k}" mass="{mass:.6f}" '
                     f'rgba="0.25 0.35 0.75 1" {ROBOT_GEOM}/>')
    return "\n    ".join(assets), "\n      ".join(geoms)


def _frame_attrs(morph: Morphology, node) -> str:
    off, rot = morph.rel_frame(node)
    return f'pos="{_f(*off)}" xyaxes="{_f(*rot[:, 0], *rot[:, 1])}"'


def _servo_box() -> str:
    half = _f(*(np.array(SERVO_BOX) / 2))
    return f'<geom type="box" size="{half}" {SERVO_GEOM}/>'


# Look only: no mass, no contact, render group 1. They show the printed
# bracket around the servo and the horn on its output face, which the physics
# approximates with the servo box and a link starting at the joint axis.
VISUAL_GEOM = 'contype="0" conaffinity="0" group="1" density="0"'
BRACKET_RGBA = "0.55 0.60 0.70 1"
HORN_RGBA = "0.80 0.80 0.82 1"
HORN_RADIUS, HORN_THICK = 0.011, 0.003


def _servo_dress(axis_local) -> str:
    """Bracket around the lower 60% of the servo and the horn on the face the
    joint axis points out of."""
    sx, sy, sz = np.array(SERVO_BOX) / 2
    wall = 0.0025
    h = 0.6 * sz
    bracket = (f'<geom type="box" pos="{_f(0, 0, -sz + h - wall)}" '
               f'size="{_f(sx + wall, sy + wall, h)}" rgba="{BRACKET_RGBA}" {VISUAL_GEOM}/>')
    a = np.abs(np.asarray(axis_local, dtype=float))
    face = float(a @ np.array([sx, sy, sz]))
    start, end = a * face, a * (face + HORN_THICK)
    horn = (f'<geom type="cylinder" fromto="{_f(*start, *end)}" size="{HORN_RADIUS}" '
            f'rgba="{HORN_RGBA}" {VISUAL_GEOM}/>')
    return f"{bracket}\n{horn}"


def _link_geom(node, rgba: str = LINK_RGBA) -> str:
    return (f'<geom type="box" pos="{_f(node.length / 2, 0, 0)}" '
            f'size="{_f(node.length / 2, node.width / 2, plate_thick(node) / 2)}" '
            f'density="{LINK_DENSITY}" rgba="{rgba}" {ROBOT_GEOM}/>')


def _elbow_geom(parent, child, slot: int) -> str:
    """Printed elbow plate from the link tip to the side slot `slot` (1 left,
    2 right), where the child module is mounted."""
    side = 1.0 if slot == 1 else -1.0
    d = elbow_distance(child)
    return (f'<geom name="elbow{child.id}" type="box" '
            f'pos="{_f(parent.length, side * d / 2, 0)}" '
            f'size="{_f(parent.width / 2, d / 2, plate_thick(parent) / 2)}" '
            f'density="{LINK_DENSITY}" rgba="{LINK_RGBA}" {ROBOT_GEOM}/>')


def _node_xml(morph: Morphology, node, frames, children, actuators: list,
              layout: ActuatorLayout, friction: float = 1.0) -> str:
    """MJCF body of one module and, recursively, of everything below it."""
    i = node.id
    inner = ""
    if node.kind == "joint":
        name = f"j{i}"
        rng = JOINT_RANGE[node.axis]
        sign = -1.0 if node.flip else 1.0
        axis = sign * np.array(AXIS_LOCAL[node.axis])
        actuators.append(
            f'<position name="{name}" joint="{name}" kp="{JOINT_KP}" kv="0" '
            f'ctrlrange="{-rng} {rng}" forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        layout.append(Actuator(i, "joint", node.axis))
        inner = (f'<joint name="{name}" type="hinge" axis="{_f(*axis)}" '
                 f'range="{-rng} {rng}"/>\n{_servo_box()}\n{_servo_dress(AXIS_LOCAL[node.axis])}\n'
                 f'{_link_geom(node)}')
    elif node.kind == "link":
        inner = _link_geom(node)
    elif node.kind == "spring":
        sign = -1.0 if node.flip else 1.0
        axis = sign * np.array(AXIS_LOCAL[node.axis])
        inner = (f'<joint name="s{i}" type="hinge" axis="{_f(*axis)}" '
                 f'stiffness="{spring_stiffness(node):.8g}" '
                 f'damping="{SPRING_DAMPING}" frictionloss="0" armature="0" '
                 f'range="{-SPRING_RANGE} {SPRING_RANGE}"/>\n'
                 f'{_link_geom(node, SPRING_RGBA)}')
    elif node.kind == "wheel":
        pos, rot = frames[i]
        name = f"w{i}"
        hub = morph.wheel_hub(node, pos, rot)
        half_t = max(0.01, 0.25 * node.wheel_rad)
        axle = np.array([0.0, morph.wheel_axis_sign(rot), 0.0])
        actuators.append(
            f'<velocity name="{name}" joint="{name}" kv="{WHEEL_KV}" '
            f'ctrlrange="{-SERVO_SPEED} {SERVO_SPEED}" '
            f'forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        layout.append(Actuator(i, "wheel", "wheel"))
        inner = f"""{_servo_box()}
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
              density="{LINK_DENSITY}" rgba="{LINK_RGBA}" {ROBOT_GEOM}/>
          <body name="{name}" pos="{_f(*hub)}">
            <joint name="{name}" type="hinge" axis="{_f(*axle)}" damping="{CASTER_DAMPING}"
                   frictionloss="0" armature="0"/>
            <geom type="cylinder" fromto="{_f(0, -half_t, 0, 0, half_t, 0)}"
                  size="{node.wheel_rad:.4f}" density="{LIMB_DENSITY}"
                  rgba="0.1 0.1 0.1 1" {ROBOT_GEOM}/>
          </body>"""
    elif node.kind == "foot":
        inner = (f'<geom type="sphere" size="{node.foot_rad:.4f}" density="{LIMB_DENSITY}" '
                 f'rgba="0.2 0.2 0.2 1" {ROBOT_GEOM}/>')
    elif node.kind == "tail":
        mass = max(node.tail_mass, 1e-6)   # a zero-mass geom would be degenerate
        inner = (f'<geom name="tail{i}" type="box" pos="{_f(node.length / 2, 0, 0)}" '
                 f'size="{_f(node.length / 2, TAIL_BAR_SIZE[0] / 2, TAIL_BAR_SIZE[1] / 2)}" '
                 f'density="{LINK_DENSITY}" rgba="{LINK_RGBA}" {ROBOT_GEOM}/>\n'
                 f'        <geom name="tailtip{i}" type="sphere" '
                 f'pos="{_f(node.length, 0, 0)}" size="{TAIL_TIP_RADIUS:.4f}" '
                 f'mass="{mass:.6f}" rgba="0.15 0.15 0.15 1" {ROBOT_GEOM}/>')
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
        inner += "".join(f"\n{_elbow_geom(node, morph.nodes[c], slot)}"
                         for slot, c in sorted(children.get(i, {}).items()) if slot)
    # Actuators follow the depth first order of Morphology.order.
    below = "".join(
        _node_xml(morph, morph.nodes[c], frames, children, actuators, layout, friction)
        for _, c in sorted(children.get(i, {}).items()))
    return f"""
      <body name="n{i}" {_frame_attrs(morph, node)}>
        {inner}{below}
      </body>"""


def build_mjcf(morph: Morphology, terrain_assets: str = "",
               terrain_world: str = "", spawn_x: float = 0.0,
               ground_z: float = 0.0,
               payload: float = 0.0,
               terrain_friction: float = 1.0) -> Tuple[str, ActuatorLayout]:
    friction = terrain_friction
    actuators: list = []
    layout: ActuatorLayout = []
    frames = morph.frames()
    children = morph.child_map()
    legs = "".join(
        _node_xml(morph, morph.nodes[c], frames, children, actuators, layout, friction)
        for _, c in sorted(children.get(0, {}).items())
    )
    assert layout == morph.layout()
    z0 = ground_z + spawn_height(morph)
    actuator_xml = "\n    ".join(actuators)
    torso_assets, torso_geoms = _torso(morph)

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

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
printed. A child on a side slot of a link hangs from a printed elbow plate that
bridges the ELBOW offset, so no module floats. See docs/hardware-dados.md.

Self-collision is ON: robot geoms (contype 2, conaffinity 3) touch each other
and the terrain (1, 1). MuJoCo already skips parent/child pairs. A body whose
geoms already touch at spawn is not buildable (see sim.is_buildable).
"""

from typing import Tuple

import numpy as np

from darwinbot.controller import SERVO_SPEED, SERVO_TORQUE
from darwinbot.morphology import (AXIS_LOCAL, ELBOW, FOOT_RADIUS, LINK_RADIUS,
                                  SERVO_BOX, TORSO_HEIGHT, Actuator,
                                  ActuatorLayout, Morphology)

SHELL_DENSITY  = 300.0   # kg/m^3, printed shell with infill
LIMB_DENSITY   = 300.0
PAYLOAD_MASS   = 0.13    # kg: ESP32 + 2S LiPo + regulator
SERVO_MASS     = 0.060   # kg
ROBOT_FRICTION = "1.0 0.02 0.001"
SPAWN_CLEARANCE = 0.02   # m of air under the lowest foot at spawn

JOINT_RANGE = {"yaw": 0.9, "pitch": 1.3, "roll": 0.9}   # +/- rad
JOINT_DAMPING  = 0.56
JOINT_FRICTION = 0.068
JOINT_ARMATURE = 0.027
JOINT_KP       = 13.37
WHEEL_KV       = 2.0          # N*m per rad/s, saturates at SERVO_TORQUE
WHEEL_DAMPING  = 0.05         # wheel spins freely, unlike a position servo

ROBOT_GEOM = f'contype="2" conaffinity="3" friction="{ROBOT_FRICTION}"'
SERVO_GEOM = f'mass="{SERVO_MASS}" rgba="0.15 0.15 0.15 1" {ROBOT_GEOM}'


def _f(*values) -> str:
    return " ".join(f"{v:.4f}" for v in values)


def spawn_height(morph: Morphology) -> float:
    """Torso centre height that puts the lowest point of the body just above z = 0."""
    return -morph.lowest_z() + SPAWN_CLEARANCE


def _frame_attrs(morph: Morphology, node) -> str:
    off, rot = morph.rel_frame(node)
    return f'pos="{_f(*off)}" xyaxes="{_f(*rot[:, 0], *rot[:, 1])}"'


def _servo_box() -> str:
    half = _f(*(np.array(SERVO_BOX) / 2))
    return f'<geom type="box" size="{half}" {SERVO_GEOM}/>'


def _link_geom(length: float) -> str:
    return (f'<geom type="capsule" fromto="{_f(0, 0, 0, length, 0, 0)}" '
            f'size="{LINK_RADIUS:.4f}" density="{LIMB_DENSITY}" '
            f'rgba="0.55 0.60 0.70 1" {ROBOT_GEOM}/>')


def _elbow_geom(length: float, slot: int, child: int) -> str:
    """Printed elbow plate from the link tip to the side slot `slot` (1 left,
    2 right), where the child module is mounted."""
    side = 1.0 if slot == 1 else -1.0
    return (f'<geom name="elbow{child}" type="box" '
            f'pos="{_f(length, side * ELBOW / 2, 0)}" '
            f'size="{_f(LINK_RADIUS, ELBOW / 2, LINK_RADIUS)}" '
            f'density="{LIMB_DENSITY}" rgba="0.55 0.60 0.70 1" {ROBOT_GEOM}/>')


def _node_xml(morph: Morphology, node, frames, children, actuators: list,
              layout: ActuatorLayout) -> str:
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
                 f'range="{-rng} {rng}"/>\n{_servo_box()}\n{_link_geom(node.length)}')
    elif node.kind == "link":
        inner = _link_geom(node.length)
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
    elif node.kind == "foot":
        inner = (f'<geom type="sphere" size="{FOOT_RADIUS:.4f}" density="{LIMB_DENSITY}" '
                 f'rgba="0.2 0.2 0.2 1" {ROBOT_GEOM}/>')
    if node.kind in ("joint", "link"):
        inner += "".join(f"\n{_elbow_geom(node.length, slot, c)}"
                         for slot, c in sorted(children.get(i, {}).items()) if slot)
    # Actuators follow the depth first order of Morphology.order.
    below = "".join(
        _node_xml(morph, morph.nodes[c], frames, children, actuators, layout)
        for _, c in sorted(children.get(i, {}).items()))
    return f"""
      <body name="n{i}" {_frame_attrs(morph, node)}>
        {inner}{below}
      </body>"""


def build_mjcf(morph: Morphology, terrain_assets: str = "",
               terrain_world: str = "", spawn_x: float = 0.0,
               ground_z: float = 0.0) -> Tuple[str, ActuatorLayout]:
    actuators: list = []
    layout: ActuatorLayout = []
    frames = morph.frames()
    children = morph.child_map()
    legs = "".join(
        _node_xml(morph, morph.nodes[c], frames, children, actuators, layout)
        for _, c in sorted(children.get(0, {}).items())
    )
    assert layout == morph.layout()
    z0 = ground_z + spawn_height(morph)
    actuator_xml = "\n    ".join(actuators)

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
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {terrain_world}
    <body name="torso" pos="{_f(spawn_x, 0, z0)}">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="0 -1.8 0.9" xyaxes="1 0 0 0 0.45 0.9"/>
      <geom type="box" size="{_f(morph.body_length / 2, morph.body_width / 2, TORSO_HEIGHT / 2)}"
            density="{SHELL_DENSITY}" rgba="0.25 0.35 0.75 1" {ROBOT_GEOM}/>
      <geom type="box" size="{_f(min(0.05, morph.body_length / 2), 0.03, 0.015)}"
            mass="{PAYLOAD_MASS}" contype="0" conaffinity="0" rgba="0.2 0.6 0.2 0.3"/>
      {legs}
    </body>
  </worldbody>
  <actuator>
    {actuator_xml}
  </actuator>
</mujoco>"""
    return xml, layout

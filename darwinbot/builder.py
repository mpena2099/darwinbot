"""
builder.py
----------
Morphology + terrain -> MuJoCo MJCF string, plus the actuator layout the
controller needs to address each motor.

Frames: X forward, Y left, Z up. Every leg is built so that, at zero joint
angles, the thigh is horizontal and the shin hangs straight down. Joint axes
are mirrored between left and right so a given signal means the same motion
on both sides:
  hip_yaw   positive swings the foot forward
  hip_pitch positive lifts the thigh
  knee      positive swings the foot outward
  wheel     positive rolls the robot forward

Parts are real: every joint is a Feetech STS3215 (7.4 V) with the torque and
speed limits of its data sheet, minus a safety margin. Each servo adds a 60 g
box, the torso carries a payload (ESP32, 2S LiPo, regulator), and links are
printed. See docs/hardware-dados.md.

Self-collision is ON: robot geoms (contype 2, conaffinity 3) touch each other
and the terrain (1, 1). MuJoCo already skips parent/child pairs. A body whose
geoms already touch at spawn is not buildable (see sim.is_buildable).
"""

from typing import Tuple

import numpy as np

from darwinbot.controller import ActuatorLayout, SERVO_SPEED
from darwinbot.morphology import Morphology

TORSO_HEIGHT   = 0.06
SHELL_DENSITY  = 300.0   # kg/m^3, printed shell with infill
LIMB_DENSITY   = 300.0
PAYLOAD_MASS   = 0.13    # kg: ESP32 + 2S LiPo + regulator
SERVO_MASS     = 0.060   # kg
SERVO_BOX      = (0.045, 0.025, 0.035)   # m, length along the leg, width, height
ROBOT_FRICTION = "1.0 0.02 0.001"
SPAWN_CLEARANCE = 0.02   # m of air under the lowest foot at spawn

JOINT_RANGE = {"hip_yaw": 0.9, "hip_pitch": 1.3, "knee": 1.3}   # +/- rad
JOINT_DAMPING  = 0.56
JOINT_FRICTION = 0.068
JOINT_ARMATURE = 0.027
JOINT_KP       = 13.37
SERVO_TORQUE   = 1.91 * 0.7   # N*m: 7.4 V stall torque with a 30% margin
WHEEL_KV       = 2.0          # N*m per rad/s, saturates at SERVO_TORQUE
WHEEL_DAMPING  = 0.05         # wheel spins freely, unlike a position servo

ROBOT_GEOM = f'contype="2" conaffinity="3" friction="{ROBOT_FRICTION}"'
SERVO_GEOM = f'mass="{SERVO_MASS}" rgba="0.15 0.15 0.15 1" {ROBOT_GEOM}'


def _f(*values) -> str:
    return " ".join(f"{v:.4f}" for v in values)


def spawn_height(morph: Morphology) -> float:
    """Torso centre height that puts the longest leg just above z = 0."""
    return max(p.reach for p in morph.pairs) + SPAWN_CLEARANCE


def _leg_xml(morph: Morphology, pair, side: int,
             actuators: list, layout: ActuatorLayout) -> str:
    sign = 1.0 if side == 0 else -1.0
    tag  = f"p{pair.index}{'L' if side == 0 else 'R'}"

    hip_x = pair.attach_x * morph.body_length / 2 * 0.9
    hip_y = sign * morph.body_width / 2
    d = np.array([np.sin(pair.splay), sign * np.cos(pair.splay), 0.0])
    pitch_axis = np.array([d[1], -d[0], 0.0])        # d x z: positive lifts
    yaw_axis   = np.array([0.0, 0.0, -sign])         # positive swings forward
    knee_pos   = d * pair.thigh_len

    def joint(kind, axis):
        name = f"{tag}_{kind}"
        rng = JOINT_RANGE[kind]
        actuators.append(
            f'<position name="{name}" joint="{name}" kp="{JOINT_KP}" kv="0" '
            f'ctrlrange="{-rng} {rng}" forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        layout.append((pair.index, side, kind))
        return (f'<joint name="{name}" type="hinge" axis="{_f(*axis)}" '
                f'range="{-rng} {rng}"/>')

    # Actuators are appended in call order, which must follow JOINTS and
    # then the wheel: CPG relies on layout matching model order.
    hip_yaw   = joint("hip_yaw", yaw_axis) if pair.has_yaw else ""
    hip_pitch = joint("hip_pitch", pitch_axis)
    knee      = joint("knee", pitch_axis)

    # Servo boxes are long along the leg: rotate them with the splay.
    half = _f(SERVO_BOX[1] / 2, SERVO_BOX[0] / 2, SERVO_BOX[2] / 2)
    hip_box = (f'<geom type="box" size="{half}" pos="{_f(*(d * SERVO_BOX[0] / 2))}" '
               f'euler="0 0 {-sign * pair.splay:.4f}" {SERVO_GEOM}/>')
    if pair.has_yaw:   # second servo, same place: mass only matters
        hip_box += hip_box.replace('<geom ', '<geom name="%s_yaw_servo" ' % tag)
    knee_box = (f'<geom type="box" size="{half}" euler="0 0 {-sign * pair.splay:.4f}" '
                f'{SERVO_GEOM}/>')

    wheel_xml = ""
    if pair.has_wheel:
        half_t = max(0.01, 0.25 * pair.wheel_rad)
        # Servo at the shin tip, wheel hub on its outer side.
        wy = sign * (SERVO_BOX[1] / 2 + half_t)
        name = f"{tag}_wheel"
        actuators.append(
            f'<velocity name="{name}" joint="{name}" kv="{WHEEL_KV}" '
            f'ctrlrange="{-SERVO_SPEED} {SERVO_SPEED}" '
            f'forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        layout.append((pair.index, side, "wheel"))
        knee_box += (f'<geom type="box" size="{_f(*(np.array(SERVO_BOX) / 2))}" '
                     f'pos="{_f(0, 0, -pair.shin_len)}" {SERVO_GEOM}/>')
        wheel_xml = f"""
          <body name="{name}" pos="{_f(0, wy, -pair.shin_len)}">
            <joint name="{name}" type="hinge" axis="0 1 0" damping="{WHEEL_DAMPING}"/>
            <geom type="cylinder" fromto="{_f(0, -half_t, 0, 0, half_t, 0)}"
                  size="{pair.wheel_rad:.4f}" density="{LIMB_DENSITY}"
                  rgba="0.1 0.1 0.1 1" {ROBOT_GEOM}/>
          </body>"""

    return f"""
      <body name="{tag}_hip" pos="{_f(hip_x, hip_y, 0)}">
        {hip_yaw}
        {hip_pitch}
        {hip_box}
        <geom type="capsule" fromto="{_f(0, 0, 0, *knee_pos)}" size="{pair.radius:.4f}"
              density="{LIMB_DENSITY}" rgba="0.55 0.60 0.70 1" {ROBOT_GEOM}/>
        <body name="{tag}_knee" pos="{_f(*knee_pos)}">
          {knee}
          {knee_box}
          <geom type="capsule" fromto="{_f(0, 0, 0, 0, 0, -pair.shin_len)}"
                size="{pair.radius:.4f}" density="{LIMB_DENSITY}"
                rgba="0.70 0.72 0.78 1" {ROBOT_GEOM}/>{wheel_xml}
        </body>
      </body>"""


def build_mjcf(morph: Morphology, terrain_assets: str = "",
               terrain_world: str = "", spawn_x: float = 0.0,
               ground_z: float = 0.0) -> Tuple[str, ActuatorLayout]:
    actuators: list = []
    layout: ActuatorLayout = []
    legs = "".join(
        _leg_xml(morph, pair, side, actuators, layout)
        for pair in morph.pairs for side in (0, 1)
    )
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

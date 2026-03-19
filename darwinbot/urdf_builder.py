"""
urdf_builder.py
---------------
Converts a RobotGenome into a URDF XML string that PyBullet can load.

Structure produced:
  base_link  (box body, centre of the robot)
    └── limb_0_link  (cylinder)
          └── wheel_0_link  (cylinder, if has_wheel)
    └── limb_1_link
          └── wheel_1_link
    ...

Joints:
  - base → limb  : revolute (controlled by oscillator)
  - limb → wheel : continuous (free-spinning or motorised)
"""

import os
import tempfile
import numpy as np
from darwinbot.genome import RobotGenome, LimbGenes

# Body dimensions (fixed)
BODY_SIZE_X = 0.20   # m
BODY_SIZE_Y = 0.15   # m
BODY_SIZE_Z = 0.08   # m
BODY_MASS   = 1.5    # kg

LIMB_MASS_PER_M  = 0.4   # kg per metre of limb length
WHEEL_MASS       = 0.08  # kg


# ── Inertia helpers ───────────────────────────────────────────────────────────

def _box_inertia(mass, x, y, z):
    ixx = mass / 12 * (y**2 + z**2)
    iyy = mass / 12 * (x**2 + z**2)
    izz = mass / 12 * (x**2 + y**2)
    return ixx, iyy, izz


def _cylinder_inertia(mass, radius, length):
    ixx = mass / 12 * (3 * radius**2 + length**2)
    iyy = ixx
    izz = mass / 2 * radius**2
    return ixx, iyy, izz


# ── URDF XML builders ─────────────────────────────────────────────────────────

def _material(name, r, g, b, a=1.0):
    return f"""
  <material name="{name}">
    <color rgba="{r} {g} {b} {a}"/>
  </material>"""


def _inertial(mass, ixx, iyy, izz, xyz="0 0 0"):
    return f"""
    <inertial>
      <origin xyz="{xyz}" rpy="0 0 0"/>
      <mass value="{mass:.4f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0"
               iyy="{iyy:.6f}" iyz="0"
               izz="{izz:.6f}"/>
    </inertial>"""


def _base_link():
    x, y, z = BODY_SIZE_X, BODY_SIZE_Y, BODY_SIZE_Z
    ixx, iyy, izz = _box_inertia(BODY_MASS, x, y, z)
    return f"""
  <link name="base_link">
    {_inertial(BODY_MASS, ixx, iyy, izz)}
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{x} {y} {z}"/></geometry>
      <material name="body_mat"><color rgba="0.3 0.3 0.8 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{x} {y} {z}"/></geometry>
    </collision>
  </link>"""


def _limb_link(limb_idx: int, limb: LimbGenes):
    length = limb.length
    radius = limb.width / 2
    mass   = LIMB_MASS_PER_M * length
    ixx, iyy, izz = _cylinder_inertia(mass, radius, length)

    # Cylinder axis along Z by default in URDF; we orient along X via rpy
    return f"""
  <link name="limb_{limb_idx}_link">
    {_inertial(mass, ixx, iyy, izz)}
    <visual>
      <origin xyz="{length/2:.4f} 0 0" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>
      <material name="limb_mat"><color rgba="0.6 0.6 0.6 1"/></material>
    </visual>
    <collision>
      <origin xyz="{length/2:.4f} 0 0" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>
    </collision>
  </link>"""


def _wheel_link(limb_idx: int, limb: LimbGenes):
    r    = limb.wheel_rad
    mass = WHEEL_MASS
    ixx, iyy, izz = _cylinder_inertia(mass, r, r * 0.3)

    return f"""
  <link name="wheel_{limb_idx}_link">
    {_inertial(mass, ixx, iyy, izz)}
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="{r:.4f}" length="{r*0.3:.4f}"/></geometry>
      <material name="wheel_mat"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="{r:.4f}" length="{r*0.3:.4f}"/></geometry>
    </collision>
  </link>"""


def _base_to_limb_joint(limb_idx: int, limb: LimbGenes, n_limbs: int):
    """
    Revolute joint connecting base_link to limb.
    Limbs are spaced symmetrically around the body using 2π/n_limbs spacing.
    """
    angle_offset = limb_idx * (2 * np.pi / n_limbs)
    bx = BODY_SIZE_X / 2 * np.cos(angle_offset)
    by = BODY_SIZE_Y / 2 * np.sin(angle_offset)
    bz = -BODY_SIZE_Z / 4

    insertion = limb.angle

    return f"""
  <joint name="base_limb_{limb_idx}_joint" type="revolute">
    <parent link="base_link"/>
    <child  link="limb_{limb_idx}_link"/>
    <origin xyz="{bx:.4f} {by:.4f} {bz:.4f}" rpy="0 0 {insertion:.4f}"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5708" upper="1.5708" effort="10.0" velocity="5.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>"""


def _limb_to_wheel_joint(limb_idx: int, limb: LimbGenes):
    """Continuous joint at wheel_pos fraction along the limb."""
    wx = limb.length * limb.wheel_pos
    return f"""
  <joint name="limb_{limb_idx}_wheel_joint" type="continuous">
    <parent link="limb_{limb_idx}_link"/>
    <child  link="wheel_{limb_idx}_link"/>
    <origin xyz="{wx:.4f} 0 {-limb.wheel_rad:.4f}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.05"/>
  </joint>"""


# ── Main builder ──────────────────────────────────────────────────────────────

def genome_to_urdf(robot: RobotGenome) -> str:
    """Return a complete URDF XML string for the given RobotGenome."""
    parts = ['<?xml version="1.0"?>', '<robot name="darwinbot">']

    # Base body
    parts.append(_base_link())

    # Limbs (and optional wheels)
    for i, limb in enumerate(robot.limbs):
        parts.append(_limb_link(i, limb))
        parts.append(_base_to_limb_joint(i, limb, robot.n_limbs))

        if limb.has_wheel:
            parts.append(_wheel_link(i, limb))
            parts.append(_limb_to_wheel_joint(i, limb))

    parts.append("</robot>")
    return "\n".join(parts)


def write_urdf(robot: RobotGenome, path: str = None) -> str:
    """
    Write the URDF to a file and return the path.
    If path is None, writes to a temp file.
    """
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".urdf", prefix="darwinbot_")
        os.close(fd)

    urdf_str = genome_to_urdf(robot)
    with open(path, "w") as f:
        f.write(urdf_str)
    return path

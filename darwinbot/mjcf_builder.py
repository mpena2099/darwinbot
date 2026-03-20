"""
mjcf_builder.py
---------------
Converts a RobotGenome into a MuJoCo MJCF XML string.

MJCF is MuJoCo's native format. It handles contact, friction,
and joint dynamics far better than URDF.

Robot structure
---------------
worldbody
  └── base (box, floating — 6 DoF via freejoint)
        ├── limb_0 (capsule)
        │     └── wheel_0 (cylinder, if has_wheel)  ← hinge joint
        ├── limb_1
        ...
        └── limb_5

Each limb attaches to the base via a hinge joint (1 DoF, revolute).
Each wheel attaches to the limb tip via a hinge joint (spins freely).

Actuators
---------
- Limb joints  : position actuator (kp gain, driven by oscillator)
- Wheel joints : velocity actuator (driven at constant speed)
"""

import os
import tempfile
import numpy as np
from darwinbot.genome import RobotGenome, LimbGenes

# ── Body geometry constants ───────────────────────────────────────────────────
BODY_SIZE_X  = 0.20
BODY_SIZE_Y  = 0.15
BODY_SIZE_Z  = 0.08
BODY_MASS    = 1.5

LIMB_DENSITY = 400.0   # kg/m³  (similar to light aluminium alloy)
WHEEL_MASS   = 0.08

# Contact / friction defaults
FRICTION         = "1.2 0.05 0.001"   # slide, spin, roll
WHEEL_FRICTION   = "1.5 0.05 0.001"
SOLIMP           = "0.99 0.99 0.01"
SOLREF           = "0.01 1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rgba_limb(idx: int) -> str:
    """Give each limb a slightly different grey shade for visibility."""
    v = 0.55 + 0.07 * (idx % 3)
    return f"{v:.2f} {v:.2f} {v:.2f} 1"


def _limb_attachment(limb_idx: int, n_limbs: int) -> np.ndarray:
    """
    XYZ position where limb_idx attaches to the base body.
    Evenly distributed around the body perimeter.
    """
    angle = limb_idx * (2 * np.pi / n_limbs)
    x = BODY_SIZE_X / 2 * np.cos(angle)
    y = BODY_SIZE_Y / 2 * np.sin(angle)
    z = -BODY_SIZE_Z / 4
    return np.array([x, y, z])


# ── MJCF builder ──────────────────────────────────────────────────────────────

def genome_to_mjcf(robot: RobotGenome) -> str:
    """Return a complete MJCF XML string for the given RobotGenome."""

    n_limbs = robot.n_limbs
    limb_bodies   = []
    actuator_lines = []

    for i, limb in enumerate(robot.limbs):
        attach = _limb_attachment(i, n_limbs)
        ax, ay, az = attach

        # Limb axis direction (unit vector from base outward)
        angle  = limb_idx_angle = i * (2 * np.pi / n_limbs)
        dir_x  = np.cos(angle)
        dir_y  = np.sin(angle)

        # Capsule half-length and radius
        half_l = limb.length / 2
        radius = limb.width / 2

        # Limb centre position relative to attachment point
        cx = dir_x * half_l
        cy = dir_y * half_l

        # Hinge joint axis is perpendicular to the limb direction in XY plane
        # → rotate 90° around Z
        jax = -dir_y
        jay =  dir_x

        # Mass from density × volume of capsule — clamp to minimum
        vol  = np.pi * radius**2 * limb.length + (4/3) * np.pi * radius**3
        mass = max(LIMB_DENSITY * vol, 0.05)   # minimum 50g per limb

        wheel_xml = ""
        if limb.has_wheel:
            wr   = limb.wheel_rad
            wpos = limb.wheel_pos          # fraction along limb
            # Position along limb from attachment
            wpx  = dir_x * (limb.length * wpos - half_l)
            wpy  = dir_y * (limb.length * wpos - half_l)
            wpz  = -wr                     # wheel hangs below limb

            wheel_xml = f"""
          <body name="wheel_{i}" pos="{wpx:.4f} {wpy:.4f} {wpz:.4f}">
            <joint name="wheel_{i}_joint" type="hinge"
                   axis="{-dir_y:.4f} {dir_x:.4f} 0"
                   damping="0.05" frictionloss="0.02"/>
            <geom name="wheel_{i}_geom" type="cylinder"
                  size="{wr:.4f} {wr*0.15:.4f}"
                  mass="{WHEEL_MASS:.4f}"
                  friction="{WHEEL_FRICTION}"
                  solimp="{SOLIMP}" solref="{SOLREF}"
                  rgba="0.1 0.1 0.1 1"
                  euler="90 0 0"/>
          </body>"""

            actuator_lines.append(
                f'  <velocity name="wheel_{i}_vel" joint="wheel_{i}_joint"'
                f' gear="1" kv="2.0"/>'
            )

        limb_bodies.append(f"""
      <body name="limb_{i}" pos="{ax:.4f} {ay:.4f} {az:.4f}">
        <joint name="limb_{i}_joint" type="hinge"
               axis="{jax:.4f} {jay:.4f} 0"
               range="-1.5708 1.5708"
               damping="0.5" frictionloss="0.1"
               armature="0.01"/>
        <geom name="limb_{i}_geom" type="capsule"
              fromto="{-cx:.4f} {-cy:.4f} 0  {cx:.4f} {cy:.4f} 0"
              size="{radius:.4f}"
              mass="{mass:.4f}"
              friction="{FRICTION}"
              solimp="{SOLIMP}" solref="{SOLREF}"
              rgba="{_rgba_limb(i)}"/>{wheel_xml}
      </body>""")

        actuator_lines.append(
            f'  <position name="limb_{i}_pos" joint="limb_{i}_joint"'
            f' gear="1" kp="8.0" forcelimited="true" forcerange="-10 10"/>'
        )

    limb_xml      = "\n".join(limb_bodies)
    actuator_xml  = "\n".join(actuator_lines)

    mjcf = f"""<mujoco model="darwinbot">

  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.002" iterations="100" solver="Newton"
          integrator="RK4" gravity="0 0 -9.81"
          cone="elliptic" noslip_iterations="5"/>

  <default>
    <geom condim="6" solimp="0.99 0.99 0.01" solref="0.02 1"/>
    <joint damping="0.5" armature="0.05" frictionloss="0.1"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             width="512" height="512"
             rgb1="0.85 0.85 0.85" rgb2="0.70 0.70 0.70"/>
    <material name="floor_mat" texture="grid"
              texrepeat="4 4" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light name="sun" pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="20 20 0.1"
          friction="{FRICTION}" material="floor_mat"/>
    {{scenario_xml}}

    <body name="base" pos="0 0 0.35">
      <freejoint name="root"/>
      <geom name="base_geom" type="box"
            size="{BODY_SIZE_X/2:.4f} {BODY_SIZE_Y/2:.4f} {BODY_SIZE_Z/2:.4f}"
            mass="{BODY_MASS:.4f}"
            friction="{FRICTION}"
            rgba="0.3 0.3 0.8 1"/>
      <inertial pos="0 0 0" mass="{BODY_MASS:.4f}"
                diaginertia="{BODY_MASS/12*(BODY_SIZE_Y**2+BODY_SIZE_Z**2):.6f} {BODY_MASS/12*(BODY_SIZE_X**2+BODY_SIZE_Z**2):.6f} {BODY_MASS/12*(BODY_SIZE_X**2+BODY_SIZE_Y**2):.6f}"/>
{limb_xml}
    </body>
  </worldbody>

  <actuator>
{actuator_xml}
  </actuator>

</mujoco>"""

    return mjcf


def write_mjcf(robot: RobotGenome, path: str = None,
               scenario_xml: str = "") -> str:
    """Write MJCF to file and return path. Uses temp file if path is None."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".xml", prefix="darwinbot_")
        os.close(fd)

    xml = genome_to_mjcf(robot)
    xml = xml.replace("{scenario_xml}", scenario_xml)

    with open(path, "w") as f:
        f.write(xml)
    return path

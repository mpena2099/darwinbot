"""
mjcf_builder.py
---------------
Converts a RobotGenome into MuJoCo MJCF XML.

Each limb is a chain of N_SEGMENTS capsules connected by hinge joints,
with an optional wheel cylinder at the tip of the last segment.

MJCF body hierarchy per limb
-----------------------------
base
  └── limb_i_s0  (segment 0, attached to base)
        └── limb_i_s1  (segment 1, attached to tip of s0)
              └── limb_i_s2  (segment 2, attached to tip of s1)
                    └── wheel_i  (if has_wheel, attached to tip of s2)

All joints in a limb rotate around the same axis (perpendicular to the
radial outward direction), so the limb bends in the radial plane.
Independent oscillator phases create wave-like motion along the chain.

Actuator order (important — must match _apply_control in simulator.py)
-----------------------------------------------------------------------
For each limb i (0..N_LIMBS-1):
  For each segment j (0..N_SEGMENTS-1):
    position actuator → limb_i_sj_joint
  If has_wheel:
    velocity actuator → wheel_i_joint
"""

import os
import tempfile
import numpy as np

from darwinbot.genome import RobotGenome, LimbGenes, SegmentGenes, N_SEGMENTS

# ── Body constants ────────────────────────────────────────────────────────────
BODY_SIZE_X = 0.20
BODY_SIZE_Y = 0.15
BODY_SIZE_Z = 0.08
BODY_MASS   = 1.5

LIMB_DENSITY = 400.0   # kg/m³
WHEEL_MASS   = 0.08

FRICTION       = "1.2 0.05 0.001"
WHEEL_FRICTION = "1.5 0.05 0.001"
SOLIMP         = "0.99 0.99 0.01"
SOLREF         = "0.02 1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rgba_segment(limb_idx: int, seg_idx: int) -> str:
    """Colour varies by limb (hue) and segment (brightness)."""
    hues = [
        (0.55, 0.55, 0.62),  # blue-grey
        (0.60, 0.55, 0.50),  # warm grey
        (0.50, 0.60, 0.55),  # green-grey
        (0.58, 0.52, 0.60),  # purple-grey
        (0.60, 0.58, 0.50),  # amber-grey
        (0.52, 0.58, 0.62),  # teal-grey
    ]
    base = hues[limb_idx % len(hues)]
    # Proximal segment darker, distal lighter
    brightness = 0.85 + 0.08 * seg_idx
    r = min(base[0] * brightness, 1.0)
    g = min(base[1] * brightness, 1.0)
    b = min(base[2] * brightness, 1.0)
    return f"{r:.2f} {g:.2f} {b:.2f} 1"


def _attachment_point(limb_idx: int, n_limbs: int) -> np.ndarray:
    """XYZ where limb attaches to base body perimeter."""
    angle = limb_idx * (2 * np.pi / n_limbs)
    x = BODY_SIZE_X / 2 * np.cos(angle)
    y = BODY_SIZE_Y / 2 * np.sin(angle)
    z = -BODY_SIZE_Z / 4
    return np.array([x, y, z])


def _seg_mass(seg: SegmentGenes) -> float:
    """Mass of one capsule segment, clamped to minimum."""
    r   = seg.width / 2
    vol = np.pi * r**2 * seg.length + (4/3) * np.pi * r**3
    return max(LIMB_DENSITY * vol, 0.02)


# ── Recursive segment XML builder ────────────────────────────────────────────

def _segment_chain_xml(limb_idx: int, seg_idx: int,
                       limb: LimbGenes,
                       dir_x: float, dir_y: float,
                       jax: float, jay: float,
                       actuator_lines: list) -> str:
    """
    Recursively build the XML for segment seg_idx and all its children.

    Parameters
    ----------
    limb_idx       : limb index (for naming)
    seg_idx        : segment index within limb
    limb           : LimbGenes dataclass
    dir_x, dir_y   : outward unit vector of the limb in XY plane
    jax, jay       : joint rotation axis (perpendicular to limb direction)
    actuator_lines : list to append actuator XML strings to
    """
    seg   = limb.segments[seg_idx]
    name  = f"limb_{limb_idx}_s{seg_idx}"
    mass  = _seg_mass(seg)
    r     = seg.width / 2
    l     = seg.length

    # Geom goes from body origin (proximal end) along outward direction
    tip_x = dir_x * l
    tip_y = dir_y * l

    # Register position actuator for this joint
    actuator_lines.append(
        f'  <position name="{name}_pos" joint="{name}_joint"'
        f' gear="1" kp="6.0" forcelimited="true" forcerange="-8 8"/>'
    )

    # Build children (next segment or wheel)
    children_xml = ""

    if seg_idx < N_SEGMENTS - 1:
        # Next segment: body positioned at tip of current segment
        next_xml = _segment_chain_xml(
            limb_idx, seg_idx + 1, limb,
            dir_x, dir_y, jax, jay,
            actuator_lines,
        )
        children_xml += f"""
      <body name="limb_{limb_idx}_s{seg_idx+1}" pos="{tip_x:.4f} {tip_y:.4f} 0">
        <joint name="limb_{limb_idx}_s{seg_idx+1}_joint" type="hinge"
               axis="{jax:.4f} {jay:.4f} 0"
               range="-1.5708 1.5708"
               damping="0.4" frictionloss="0.08" armature="0.01"/>
{next_xml}
      </body>"""

    else:
        # Last segment — attach wheel if present
        if limb.has_wheel:
            wr         = limb.wheel_rad
            wthick     = wr * 0.20     # half-thickness of wheel (thin like a bicycle wheel)
            axle_half  = wr * 0.65     # half-length of axle (sticks out on both sides)
            axle_r     = wr * 0.07     # axle radius (thin cylinder)

            # The wheel body centre is placed offset along the axle direction
            # from the segment tip, so that one end of the axle coincides with
            # the segment tip — exactly like a bicycle axle held at one end.
            #
            # Axle direction = (jax, jay, 0) — perpendicular to limb in XY plane.
            # The wheel face is perpendicular to the axle, so the wheel rolls
            # in the limb's outward direction (forward locomotion).
            #
            # Body centre = segment_tip + axle_half * axle_dir + drop_z
            cx = tip_x + jax * axle_half
            cy = tip_y + jay * axle_half
            cz = -(r + wthick)   # drop so wheel sits below segment tip

            # fromto in local body frame for axle and wheel:
            # axle: from -axle_half along axle_dir to +axle_half
            ax0x = -jax * axle_half;  ax0y = -jay * axle_half
            ax1x =  jax * axle_half;  ax1y =  jay * axle_half
            # wheel: same axis direction but only the thickness extent
            wx0x = -jax * wthick;  wx0y = -jay * wthick
            wx1x =  jax * wthick;  wx1y =  jay * wthick

            children_xml += f"""
      <body name="wheel_{limb_idx}" pos="{cx:.4f} {cy:.4f} {cz:.4f}">
        <joint name="wheel_{limb_idx}_joint" type="hinge"
               axis="{jax:.4f} {jay:.4f} 0"
               damping="0.05" frictionloss="0.02"/>
        <geom name="axle_{limb_idx}_geom" type="cylinder"
              fromto="{ax0x:.4f} {ax0y:.4f} 0  {ax1x:.4f} {ax1y:.4f} 0"
              size="{axle_r:.4f}"
              mass="0.01"
              rgba="0.65 0.65 0.65 1"/>
        <geom name="wheel_{limb_idx}_geom" type="cylinder"
              fromto="{wx0x:.4f} {wx0y:.4f} 0  {wx1x:.4f} {wx1y:.4f} 0"
              size="{wr:.4f}"
              mass="{WHEEL_MASS:.4f}"
              friction="{WHEEL_FRICTION}"
              solimp="{SOLIMP}" solref="{SOLREF}"
              rgba="0.1 0.1 0.1 1"/>
      </body>"""
            actuator_lines.append(
                f'  <velocity name="wheel_{limb_idx}_vel"'
                f' joint="wheel_{limb_idx}_joint" gear="1" kv="2.0"/>'
            )

    return f"""        <geom name="{name}_geom" type="capsule"
              fromto="0 0 0  {tip_x:.4f} {tip_y:.4f} 0"
              size="{r:.4f}"
              mass="{mass:.4f}"
              friction="{FRICTION}"
              solimp="{SOLIMP}" solref="{SOLREF}"
              rgba="{_rgba_segment(limb_idx, seg_idx)}"/>{children_xml}"""


# ── Main MJCF builder ─────────────────────────────────────────────────────────

def genome_to_mjcf(robot: RobotGenome) -> str:
    """Return a complete MJCF XML string for the given RobotGenome."""

    n_limbs        = robot.n_limbs
    limb_bodies    = []
    actuator_lines = []

    for i, limb in enumerate(robot.limbs):
        attach = _attachment_point(i, n_limbs)
        ax, ay, az = attach

        # Global outward direction for this limb
        global_angle = i * (2 * np.pi / n_limbs)
        dir_x =  np.cos(global_angle)
        dir_y =  np.sin(global_angle)

        # Joint axis perpendicular to limb direction in XY plane
        jax = -dir_y
        jay =  dir_x

        # Insertion angle gene rotates the joint range offset
        insertion = limb.angle

        # Build segment chain XML (populates actuator_lines)
        chain_xml = _segment_chain_xml(
            i, 0, limb, dir_x, dir_y, jax, jay, actuator_lines
        )

        limb_bodies.append(f"""
      <body name="limb_{i}_s0" pos="{ax:.4f} {ay:.4f} {az:.4f}">
        <joint name="limb_{i}_s0_joint" type="hinge"
               axis="{jax:.4f} {jay:.4f} 0"
               range="-1.5708 1.5708"
               ref="{insertion:.4f}"
               damping="0.5" frictionloss="0.1" armature="0.01"/>
{chain_xml}
      </body>""")

    limb_xml     = "\n".join(limb_bodies)
    actuator_xml = "\n".join(actuator_lines)

    mjcf = f"""<mujoco model="darwinbot">

  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.002" iterations="100" solver="Newton"
          integrator="RK4" gravity="0 0 -9.81"
          cone="elliptic" noslip_iterations="5"/>

  <default>
    <geom condim="6" solimp="{SOLIMP}" solref="{SOLREF}"/>
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

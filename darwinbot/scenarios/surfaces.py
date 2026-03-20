"""
scenarios/surfaces.py
---------------------
Phase 2 — three surface zones along the X axis.

  X=0..3  : smooth wood   (low friction)
  X=3..6  : rough concrete (high friction)
  X=6..9  : carpet         (very high friction)

In MuJoCo, friction is a geom property set at compile time via MJCF.
Each zone is a separate box geom with its own friction values.
"""

from darwinbot.scenarios import ScenarioInfo

ZONE_W = 4.0    # Y width of each zone
ZONE_T = 0.05   # thickness


def get_xml() -> str:
    """MJCF worldbody snippet: three floor zones with different friction."""
    return f"""
    <!-- Wood: low friction -->
    <geom name="zone_wood" type="box"
          pos="1.5 0 -{ZONE_T/2:.3f}" size="1.5 {ZONE_W/2:.1f} {ZONE_T/2:.3f}"
          friction="0.35 0.02 0.001"
          rgba="0.72 0.52 0.30 1" condim="6"/>

    <!-- Concrete: high friction -->
    <geom name="zone_concrete" type="box"
          pos="4.5 0 -{ZONE_T/2:.3f}" size="1.5 {ZONE_W/2:.1f} {ZONE_T/2:.3f}"
          friction="1.20 0.10 0.005"
          rgba="0.55 0.55 0.55 1" condim="6"/>

    <!-- Carpet: very high friction -->
    <geom name="zone_carpet" type="box"
          pos="7.5 0 -{ZONE_T/2:.3f}" size="1.5 {ZONE_W/2:.1f} {ZONE_T/2:.3f}"
          friction="2.50 0.30 0.010"
          rgba="0.40 0.20 0.50 1" condim="6"/>

    <!-- Side walls to keep robot on track -->
    <geom name="wall_left" type="box"
          pos="4.5 {ZONE_W/2+0.05:.2f} 0.25" size="4.5 0.05 0.3"
          rgba="0.8 0.8 0.8 0.3" contype="1" conaffinity="1"/>
    <geom name="wall_right" type="box"
          pos="4.5 -{ZONE_W/2+0.05:.2f} 0.25" size="4.5 0.05 0.3"
          rgba="0.8 0.8 0.8 0.3" contype="1" conaffinity="1"/>
"""


def build(model, data) -> ScenarioInfo:
    return ScenarioInfo(
        name="surfaces",
        spawn_pos=(0.0, 0.0, 0.35),
        ground_friction=1.35,   # rough average
    )

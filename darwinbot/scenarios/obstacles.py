"""
scenarios/obstacles.py
----------------------
Phase 3 — step, staircase, ramp in sequence along X axis.

  X=0..1.5  : flat approach
  X=1.5     : single step   0.06m high
  X=2.0..4  : flat landing
  X=4..5.2  : staircase     3 steps × 0.05m rise, 0.40m run
  X=5.2..7  : flat landing
  X=7..9    : ramp           0.12m rise over 2.0m (~3.4°)
  X=9..10   : flat top
"""

import numpy as np
from darwinbot.scenarios import ScenarioInfo

FLOOR_W = 2.0   # half-width in Y
SLAB_T  = 0.05  # floor slab thickness


def get_xml() -> str:
    stair_rise = 0.05
    stair_run  = 0.40
    step_h     = 0.06
    x_stair    = 4.0
    z_base     = step_h

    # Build stair XML
    stair_xml = ""
    for i in range(3):
        rise = stair_rise * (i + 1)
        x0   = x_stair + i * stair_run
        cx   = x0 + stair_run / 2
        cz   = (z_base + rise) / 2
        stair_xml += f"""
    <geom name="stair_{i}" type="box"
          pos="{cx:.4f} 0 {cz:.4f}"
          size="{stair_run/2:.4f} {FLOOR_W:.2f} {(z_base+rise)/2:.4f}"
          friction="0.9 0.05 0.001"
          rgba="0.75 0.65 0.55 1" condim="6"/>"""

    top_z      = z_base + stair_rise * 3
    ramp_length = 2.0
    ramp_rise   = 0.12
    ramp_angle  = np.arctan2(ramp_rise, ramp_length)
    ramp_cx     = 8.0
    ramp_cz     = top_z + ramp_rise / 2

    return f"""
    <!-- Approach floor -->
    <geom name="approach" type="box"
          pos="0.75 0 -{SLAB_T/2:.3f}" size="0.75 {FLOOR_W} {SLAB_T/2:.3f}"
          friction="0.9 0.05 0.001" rgba="0.65 0.65 0.65 1" condim="6"/>

    <!-- Single step -->
    <geom name="step" type="box"
          pos="1.75 0 {step_h/2:.4f}" size="0.5 {FLOOR_W} {step_h/2:.4f}"
          friction="0.9 0.05 0.001" rgba="0.80 0.70 0.50 1" condim="6"/>

    <!-- Landing after step -->
    <geom name="landing_1" type="box"
          pos="3.0 0 {step_h-SLAB_T/2:.4f}" size="1.0 {FLOOR_W} {SLAB_T/2:.3f}"
          friction="0.9 0.05 0.001" rgba="0.65 0.65 0.65 1" condim="6"/>

    <!-- Staircase -->
    {stair_xml}

    <!-- Landing after stairs -->
    <geom name="landing_2" type="box"
          pos="5.8 0 {top_z-SLAB_T/2:.4f}" size="0.8 {FLOOR_W} {SLAB_T/2:.3f}"
          friction="0.9 0.05 0.001" rgba="0.65 0.65 0.65 1" condim="6"/>

    <!-- Ramp -->
    <geom name="ramp" type="box"
          pos="{ramp_cx:.4f} 0 {ramp_cz:.4f}"
          size="{ramp_length/2:.4f} {FLOOR_W} 0.025"
          euler="0 {-ramp_angle:.4f} 0"
          friction="0.9 0.05 0.001"
          rgba="0.60 0.75 0.60 1" condim="6"/>

    <!-- Top landing -->
    <geom name="top" type="box"
          pos="9.5 0 {top_z+ramp_rise-SLAB_T/2:.4f}"
          size="0.5 {FLOOR_W} {SLAB_T/2:.3f}"
          friction="0.9 0.05 0.001" rgba="0.65 0.65 0.65 1" condim="6"/>

    <!-- Side walls -->
    <geom name="obs_wall_l" type="box"
          pos="5.0 {FLOOR_W+0.05:.2f} 0.3" size="5.0 0.05 0.35"
          rgba="0.7 0.7 0.7 0.3" contype="1" conaffinity="1"/>
    <geom name="obs_wall_r" type="box"
          pos="5.0 -{FLOOR_W+0.05:.2f} 0.3" size="5.0 0.05 0.35"
          rgba="0.7 0.7 0.7 0.3" contype="1" conaffinity="1"/>
"""


def build(model, data) -> ScenarioInfo:
    return ScenarioInfo(
        name="obstacles",
        spawn_pos=(0.0, 0.0, 0.35),
        ground_friction=0.9,
    )

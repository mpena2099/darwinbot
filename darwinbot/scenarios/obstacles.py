"""
scenarios/obstacles.py
----------------------
Phase 3 — vertical obstacles in sequence along the X axis.

Layout (side view, robot travels left → right):

  X=0..1.5 : flat approach      (ground level)
  X=1.5    : single step  0.06m high
  X=2.5..4 : flat landing
  X=4..5.5 : staircase    3 steps × 0.05m rise, 0.40m run each
  X=5.5..7 : flat landing
  X=7..9   : ramp         0.12m rise over 2.0m run (~3.4° slope)
  X=9..10  : flat top

All surfaces use moderate friction (concrete-like).
The robot spawns at X=0, Z=0.35 and is scored on forward distance.
A bonus is awarded for each obstacle cleared (crosses the X threshold
at or above the required height).
"""

import numpy as np
from darwinbot.scenarios import ScenarioInfo

FLOOR_FRICTION   = 0.9
OBSTACLE_FRICTION = 0.85
FLOOR_W = 2.0    # half-width in Y


def _box(half_extents, pos, rgba, client):
    import pybullet as p
    col = p.createCollisionShape(p.GEOM_BOX,
                                 halfExtents=half_extents,
                                 physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX,
                              halfExtents=half_extents,
                              rgbaColor=rgba,
                              physicsClientId=client)
    body = p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos,
                             physicsClientId=client)
    p.changeDynamics(body, -1,
                     lateralFriction=OBSTACLE_FRICTION,
                     physicsClientId=client)
    return body


def _floor_slab(x_start, length, z_top, client):
    """Flat slab from x_start to x_start+length with top surface at z_top."""
    thickness = 0.05
    _box(
        half_extents=[length / 2, FLOOR_W, thickness / 2],
        pos=[x_start + length / 2, 0.0, z_top - thickness / 2],
        rgba=[0.65, 0.65, 0.65, 1],
        client=client,
    )


def build(client: int) -> ScenarioInfo:
    import pybullet as p
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=client)

    # ── Approach floor ────────────────────────────────────────────────────────
    _floor_slab(0.0, 1.5, z_top=0.0, client=client)

    # ── Single step ───────────────────────────────────────────────────────────
    step_h = 0.06
    _box(half_extents=[0.5, FLOOR_W, step_h / 2],
         pos=[1.5 + 0.5, 0.0, step_h / 2],
         rgba=[0.80, 0.70, 0.50, 1],
         client=client)
    _floor_slab(2.0, 2.0, z_top=step_h, client=client)

    # ── Staircase (3 steps) ───────────────────────────────────────────────────
    stair_rise = 0.05
    stair_run  = 0.40
    x_stair    = 4.0
    z_base     = step_h
    for i in range(3):
        rise = stair_rise * (i + 1)
        x0   = x_stair + i * stair_run
        # Vertical riser
        _box(half_extents=[stair_run / 2, FLOOR_W, (z_base + rise) / 2],
             pos=[x0 + stair_run / 2, 0.0, (z_base + rise) / 2],
             rgba=[0.75, 0.65, 0.55, 1],
             client=client)

    top_z = z_base + stair_rise * 3
    _floor_slab(x_stair + 3 * stair_run, 1.5, z_top=top_z, client=client)

    # ── Ramp ─────────────────────────────────────────────────────────────────
    ramp_x_start = 7.0
    ramp_length  = 2.0
    ramp_rise    = 0.12
    ramp_angle   = np.arctan2(ramp_rise, ramp_length)   # ~3.4°
    ramp_cx      = ramp_x_start + ramp_length / 2
    ramp_cz      = top_z + ramp_rise / 2
    ramp_orn     = p.getQuaternionFromEuler([0, -ramp_angle, 0])

    col = p.createCollisionShape(p.GEOM_BOX,
                                 halfExtents=[ramp_length / 2, FLOOR_W, 0.025],
                                 physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX,
                              halfExtents=[ramp_length / 2, FLOOR_W, 0.025],
                              rgbaColor=[0.60, 0.75, 0.60, 1],
                              physicsClientId=client)
    ramp = p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[ramp_cx, 0.0, ramp_cz],
                             baseOrientation=ramp_orn,
                             physicsClientId=client)
    p.changeDynamics(ramp, -1,
                     lateralFriction=OBSTACLE_FRICTION,
                     physicsClientId=client)

    ramp_top_z = top_z + ramp_rise
    _floor_slab(ramp_x_start + ramp_length, 1.0, z_top=ramp_top_z,
                client=client)

    # ── Side walls (keep robot on track) ─────────────────────────────────────
    wall_col = p.createCollisionShape(p.GEOM_BOX,
                                      halfExtents=[5.0, 0.05, 0.3],
                                      physicsClientId=client)
    for y_sign in (+1, -1):
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=wall_col,
                          basePosition=[5.0, y_sign * (FLOOR_W + 0.05), 0.3],
                          physicsClientId=client)

    return ScenarioInfo(
        name="obstacles",
        spawn_pos=(0.0, 0.0, 0.35),
        ground_friction=FLOOR_FRICTION,
    )

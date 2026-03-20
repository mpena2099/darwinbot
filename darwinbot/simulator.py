"""
simulator.py
------------
MuJoCo-based simulation pipeline.
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
from typing import Callable, Optional, Tuple

from darwinbot.genome import RobotGenome, oscillator_signal, decode
from darwinbot.mjcf_builder import write_mjcf
from darwinbot.scenarios import ScenarioInfo

# ── Simulation parameters ─────────────────────────────────────────────────────
SIM_HZ      = 500          # steps per second (timestep = 0.002 s)
SIM_SECONDS = 10.0
SIM_STEPS   = int(SIM_HZ * SIM_SECONDS)

# How many steps to wait before checking for falls
# Gives the robot time to settle from the spawn drop
SETTLE_STEPS = 100


# ── Control ───────────────────────────────────────────────────────────────────

def _apply_control(model: mujoco.MjModel, data: mujoco.MjData,
                   robot: RobotGenome, t: float):
    """Write actuator targets into data.ctrl."""
    ctrl_idx = 0
    for i, limb in enumerate(robot.limbs):
        # Position control: oscillator target angle
        data.ctrl[ctrl_idx] = oscillator_signal(limb, t)
        ctrl_idx += 1
        # Velocity control: constant wheel spin
        if limb.has_wheel:
            data.ctrl[ctrl_idx] = 10.0
            ctrl_idx += 1


# ── Fall detection ────────────────────────────────────────────────────────────

def _fell(data: mujoco.MjData, spawn_z: float) -> bool:
    """
    Check if the robot has toppled.

    data.xpos[1] = base body world position (x, y, z)
    data.xmat[1] = base body rotation matrix, stored column-major (9 floats)
                   Column layout: [Xx Yx Zx | Xy Yy Zy | Xz Yz Zz]
                   The world-Z component of the body's local Z axis is xmat[1][8]
                   (last element = Zz). When upright, Zz ≈ 1.0.
    """
    base_z = data.xpos[1][2]
    zz     = data.xmat[1][8]          # cos(tilt from vertical)
    tilt   = 1.0 - zz                 # 0 = upright, 2 = upside-down

    too_low    = base_z < spawn_z - 0.20
    too_tilted = tilt > 0.8           # ~63° from vertical

    return bool(too_low or too_tilted)


# ── Main simulate function ────────────────────────────────────────────────────

def simulate(
    genes: np.ndarray,
    scenario: Optional[Tuple[Callable, Callable]] = None,
    visualise: bool = False,
) -> dict:
    """
    Simulate one individual and return performance metrics.

    Parameters
    ----------
    genes     : flat genome vector
    scenario  : (get_xml_fn, build_fn) tuple from registry.
                If None, uses flat floor.
    visualise : open interactive MuJoCo viewer (blocks until closed)

    Returns
    -------
    dict with keys:
        distance_x   – forward displacement (m)
        lateral_dev  – absolute lateral drift (m)
        fell         – True if robot toppled
        energy       – sum |ctrl × qvel| over all steps
        final_pos    – (x, y, z) tuple
        scenario     – scenario name string
        _pos0, _pos_final, _info
    """
    from darwinbot.scenarios.flat import build as flat_build, get_xml as flat_xml
    if scenario is None:
        get_xml_fn = flat_xml
        build_fn   = flat_build
    else:
        get_xml_fn, build_fn = scenario

    robot = decode(genes)
    mjcf  = write_mjcf(robot, scenario_xml=get_xml_fn())

    try:
        model = mujoco.MjModel.from_xml_path(mjcf)
        data  = mujoco.MjData(model)

        # Get scenario info (spawn position etc.)
        info: ScenarioInfo = build_fn(model, data)

        # Reset state and set spawn position
        mujoco.mj_resetData(model, data)
        data.qpos[0] = info.spawn_pos[0]   # x
        data.qpos[1] = info.spawn_pos[1]   # y
        data.qpos[2] = info.spawn_pos[2]   # z
        # qpos[3:7] = quaternion, default [1,0,0,0] = upright — leave as is
        mujoco.mj_forward(model, data)

        # Record spawn position after settling
        # Let the robot drop and settle for SETTLE_STEPS before recording pos0
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)

        pos0    = np.array(data.xpos[1]).copy()
        spawn_z = pos0[2]
        energy  = 0.0
        fell    = False

        if visualise:
            import time
            with mujoco.viewer.launch_passive(model, data) as viewer:
                step = 0
                while viewer.is_running():
                    t = step / SIM_HZ
                    _apply_control(model, data, robot, t)
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(1.0 / SIM_HZ)
                    step += 1
        else:
            for step in range(SIM_STEPS):
                t = step / SIM_HZ
                _apply_control(model, data, robot, t)
                mujoco.mj_step(model, data)

                if _fell(data, spawn_z):
                    fell = True
                    break

                # Energy proxy
                n_ctrl = model.nu
                energy += float(np.sum(
                    np.abs(data.ctrl[:n_ctrl]) *
                    np.abs(data.qvel[6:6 + n_ctrl])
                ) / SIM_HZ)

        pos_final = np.array(data.xpos[1]).copy()

        return {
            "distance_x":  float(pos_final[0] - pos0[0]),
            "lateral_dev": float(abs(pos_final[1] - pos0[1])),
            "fell":        fell,
            "energy":      float(energy),
            "final_pos":   tuple(pos_final),
            "scenario":    info.name,
            "_pos0":       tuple(pos0),
            "_pos_final":  tuple(pos_final),
            "_info":       info,
        }

    finally:
        try:
            os.remove(mjcf)
        except OSError:
            pass

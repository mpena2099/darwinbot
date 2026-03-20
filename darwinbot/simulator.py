"""
simulator.py
------------
MuJoCo-based simulation pipeline.

_apply_control is updated to drive N_SEGMENTS joints per limb,
with independent oscillator signals per segment.
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
from typing import Callable, Optional, Tuple

from darwinbot.genome import RobotGenome, segment_signal, decode
from darwinbot.mjcf_builder import write_mjcf
from darwinbot.scenarios import ScenarioInfo

# ── Simulation parameters ─────────────────────────────────────────────────────
SIM_HZ       = 500
SIM_SECONDS  = 10.0
SIM_STEPS    = int(SIM_HZ * SIM_SECONDS)
SETTLE_STEPS = 150   # steps to settle before recording pos0


# ── Control ───────────────────────────────────────────────────────────────────

def _apply_control(model: mujoco.MjModel, data: mujoco.MjData,
                   robot: RobotGenome, t: float):
    """
    Write actuator targets into data.ctrl.

    Actuator order (must match mjcf_builder.py):
      For each limb i:
        For each segment j:
          data.ctrl[idx] = position target  ← oscillator signal
        If has_wheel:
          data.ctrl[idx] = velocity target  ← constant spin
    """
    ctrl_idx = 0
    for limb in robot.limbs:
        for seg in limb.segments:
            data.ctrl[ctrl_idx] = segment_signal(seg, t)
            ctrl_idx += 1
        if limb.has_wheel:
            data.ctrl[ctrl_idx] = 10.0   # rad/s constant wheel spin
            ctrl_idx += 1


# ── Fall detection ────────────────────────────────────────────────────────────

def _fell(data: mujoco.MjData, spawn_z: float) -> bool:
    """
    Check if the robot has toppled.
    data.xmat[1] is the base body rotation matrix (row-major, 9 floats).
    xmat[8] = Zz = cosine of tilt from vertical. 1.0 = upright.
    """
    base_z = data.xpos[1][2]
    zz     = data.xmat[1][8]
    tilt   = 1.0 - zz

    too_low    = base_z < spawn_z - 0.20
    too_tilted = tilt > 0.8

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
        distance_x, lateral_dev, fell, energy,
        final_pos, scenario, _pos0, _pos_final, _info
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

        info: ScenarioInfo = build_fn(model, data)

        mujoco.mj_resetData(model, data)
        data.qpos[0] = info.spawn_pos[0]
        data.qpos[1] = info.spawn_pos[1]
        data.qpos[2] = info.spawn_pos[2]
        mujoco.mj_forward(model, data)

        # Settle — let robot drop and stabilise before measuring
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

                n_ctrl  = model.nu
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

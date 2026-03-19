"""
simulator.py
------------
Loads a URDF into PyBullet, runs the oscillator controller,
and returns raw simulation metrics.

Always runs in DIRECT mode (no GUI) for speed.
Use visualise=True only for manual inspection of a single individual.

A scenario builder function can be passed to build the world before
loading the robot. If None, defaults to the flat floor.
"""

import os
import numpy as np
import pybullet as p
import pybullet_data
from typing import Callable, Optional

from darwinbot.genome import RobotGenome, oscillator_signal, decode
from darwinbot.urdf_builder import write_urdf
from darwinbot.scenarios import ScenarioInfo

# Simulation parameters
SIM_HZ       = 120
SIM_SECONDS  = 10.0
SIM_STEPS    = int(SIM_HZ * SIM_SECONDS)


def _connect(visualise: bool) -> int:
    mode   = p.GUI if visualise else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / SIM_HZ,  physicsClientId=client)
    return client


def _load_robot(urdf_path: str, spawn_pos, spawn_orn_euler,
                client: int) -> int:
    orn = p.getQuaternionFromEuler(list(spawn_orn_euler))
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=list(spawn_pos),
        baseOrientation=orn,
        physicsClientId=client,
        flags=p.URDF_USE_SELF_COLLISION,
    )
    return robot_id


def _get_revolute_joints(robot_id: int, client: int) -> dict:
    joint_map = {}
    n = p.getNumJoints(robot_id, physicsClientId=client)
    for j in range(n):
        info   = p.getJointInfo(robot_id, j, physicsClientId=client)
        j_name = info[1].decode("utf-8")
        j_type = info[2]
        if j_type == p.JOINT_REVOLUTE:
            joint_map[j_name] = j
    return joint_map


def _apply_control(robot_id: int, joint_map: dict,
                   robot: RobotGenome, t: float, client: int):
    for i, limb in enumerate(robot.limbs):
        j_name = f"base_limb_{i}_joint"
        if j_name in joint_map:
            target = oscillator_signal(limb, t)
            p.setJointMotorControl2(
                robot_id, joint_map[j_name],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=8.0,
                physicsClientId=client,
            )
        if limb.has_wheel:
            w_name = f"limb_{i}_wheel_joint"
            n_j = p.getNumJoints(robot_id, physicsClientId=client)
            for j in range(n_j):
                info = p.getJointInfo(robot_id, j, physicsClientId=client)
                if info[1].decode("utf-8") == w_name:
                    p.setJointMotorControl2(
                        robot_id, j,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=10.0,
                        force=2.0,
                        physicsClientId=client,
                    )
                    break


def simulate(
    genes: np.ndarray,
    scenario_builder: Optional[Callable] = None,
    visualise: bool = False,
) -> dict:
    """
    Full simulation pipeline for one individual in one scenario.

    Parameters
    ----------
    genes            : flat genome vector
    scenario_builder : callable(client) -> ScenarioInfo
                       If None, uses flat floor.
    visualise        : open GUI window (manual inspection only)

    Returns
    -------
    dict with keys:
        distance_x   – forward displacement (m)
        lateral_dev  – absolute lateral drift (m)
        fell         – True if robot toppled
        energy       – sum |torque × velocity| over all steps
        final_pos    – (x, y, z) tuple
        scenario     – scenario name string
        _pos0        – spawn position tuple
        _pos_final   – final position tuple
        _info        – ScenarioInfo object
    """
    from darwinbot.scenarios.flat import build as flat_build
    if scenario_builder is None:
        scenario_builder = flat_build

    robot  = decode(genes)
    urdf   = write_urdf(robot)
    client = _connect(visualise)

    try:
        info: ScenarioInfo = scenario_builder(client)
        robot_id  = _load_robot(urdf, info.spawn_pos, info.spawn_orn, client)
        joint_map = _get_revolute_joints(robot_id, client)

        pos0 = np.array(p.getBasePositionAndOrientation(
            robot_id, physicsClientId=client)[0])

        energy = 0.0
        fell   = False

        for step in range(SIM_STEPS):
            t = step / SIM_HZ
            _apply_control(robot_id, joint_map, robot, t, client)
            p.stepSimulation(physicsClientId=client)

            pos, orn = p.getBasePositionAndOrientation(
                robot_id, physicsClientId=client)
            roll, pitch, _ = p.getEulerFromQuaternion(orn)

            if pos[2] < pos0[2] - 0.15 or abs(roll) > 1.2 or abs(pitch) > 1.2:
                fell = True
                break

            n_j = p.getNumJoints(robot_id, physicsClientId=client)
            for j in range(n_j):
                js      = p.getJointState(robot_id, j, physicsClientId=client)
                energy += abs(js[3]) * abs(js[1]) * (1.0 / SIM_HZ)

        pos_final = np.array(p.getBasePositionAndOrientation(
            robot_id, physicsClientId=client)[0])

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
        p.disconnect(physicsClientId=client)
        try:
            os.remove(urdf)
        except OSError:
            pass

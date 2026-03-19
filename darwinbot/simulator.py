"""
simulator.py
------------
Loads a URDF into PyBullet, runs the oscillator controller,
and returns raw simulation metrics.

Always runs in DIRECT mode (no GUI) for speed.
Use visualise=True only for manual inspection of a single individual.
"""

import os
import numpy as np
import pybullet as p
import pybullet_data

from darwinbot.genome import RobotGenome, oscillator_signal, decode
from darwinbot.urdf_builder import write_urdf

# Simulation parameters
SIM_HZ        = 120       # physics steps per second
SIM_SECONDS   = 10.0      # evaluation duration
SIM_STEPS     = int(SIM_HZ * SIM_SECONDS)
SPAWN_HEIGHT  = 0.35      # m above ground at spawn


def _connect(visualise: bool) -> int:
    """Connect to PyBullet and return the physics client ID."""
    mode = p.GUI if visualise else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / SIM_HZ, physicsClientId=client)
    return client


def _load_robot(urdf_path: str, client: int) -> int:
    """Load ground plane + robot, return robot body ID."""
    p.loadURDF("plane.urdf", physicsClientId=client)
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, SPAWN_HEIGHT],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=client,
        flags=p.URDF_USE_SELF_COLLISION,
    )
    return robot_id


def _get_joint_map(robot_id: int, client: int) -> dict:
    """
    Returns a dict mapping joint name → joint index
    for all revolute/continuous joints.
    """
    joint_map = {}
    n = p.getNumJoints(robot_id, physicsClientId=client)
    for j in range(n):
        info = p.getJointInfo(robot_id, j, physicsClientId=client)
        j_name = info[1].decode("utf-8")
        j_type = info[2]
        if j_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joint_map[j_name] = j
    return joint_map


def _apply_control(robot_id: int, joint_map: dict,
                   robot: RobotGenome, t: float, client: int):
    """
    Apply oscillator-driven position control to each limb joint.
    Wheel joints are driven at constant velocity (rolling).
    """
    for i, limb in enumerate(robot.limbs):
        # -- Limb revolute joint --
        j_name = f"base_limb_{i}_joint"
        if j_name in joint_map:
            target_angle = oscillator_signal(limb, t)
            p.setJointMotorControl2(
                robot_id,
                joint_map[j_name],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=8.0,
                physicsClientId=client,
            )

        # -- Wheel continuous joint (constant rolling velocity) --
        if limb.has_wheel:
            w_name = f"limb_{i}_wheel_joint"
            # find wheel joint index directly
            n = p.getNumJoints(robot_id, physicsClientId=client)
            for j in range(n):
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


def simulate(genes: np.ndarray, visualise: bool = False) -> dict:
    """
    Full simulation pipeline for one individual.

    Parameters
    ----------
    genes      : flat genome vector
    visualise  : open GUI window (for manual inspection only)

    Returns
    -------
    dict with keys:
        distance_x  – forward displacement (m)  [primary objective]
        lateral_dev – absolute lateral drift (m) [penalised]
        fell        – True if robot toppled
        energy      – sum of |torque × velocity| over all steps
        final_pos   – (x, y, z) tuple
    """
    robot  = decode(genes)
    urdf   = write_urdf(robot)
    client = _connect(visualise)

    try:
        robot_id   = _load_robot(urdf, client)
        joint_map  = _get_joint_map(robot_id, client)

        pos0 = np.array(p.getBasePositionAndOrientation(
            robot_id, physicsClientId=client)[0])

        energy = 0.0
        fell   = False

        for step in range(SIM_STEPS):
            t = step / SIM_HZ
            _apply_control(robot_id, joint_map, robot, t, client)
            p.stepSimulation(physicsClientId=client)

            # -- Check for fall --
            pos, orn = p.getBasePositionAndOrientation(
                robot_id, physicsClientId=client)
            roll, pitch, _ = p.getEulerFromQuaternion(orn)
            if pos[2] < 0.05 or abs(roll) > 1.2 or abs(pitch) > 1.2:
                fell = True
                break

            # -- Accumulate energy proxy --
            n_joints = p.getNumJoints(robot_id, physicsClientId=client)
            for j in range(n_joints):
                js = p.getJointState(robot_id, j, physicsClientId=client)
                torque   = abs(js[3])
                velocity = abs(js[1])
                energy  += torque * velocity * (1.0 / SIM_HZ)

        pos_final = np.array(p.getBasePositionAndOrientation(
            robot_id, physicsClientId=client)[0])

        return {
            "distance_x":  float(pos_final[0] - pos0[0]),
            "lateral_dev": float(abs(pos_final[1] - pos0[1])),
            "fell":        fell,
            "energy":      float(energy),
            "final_pos":   tuple(pos_final),
        }

    finally:
        p.disconnect(physicsClientId=client)
        # Clean up temp URDF
        try:
            os.remove(urdf)
        except OSError:
            pass

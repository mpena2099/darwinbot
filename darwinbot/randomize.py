"""
randomize.py
------------
Domain randomisation: perturb a compiled MuJoCo model so a gait cannot rely
on details the real robot will not reproduce (exact friction, mass, servo
strength, command latency).

The base model is never modified: `perturb` returns a copy plus the command
latency to apply. Ranges follow the tolerances in docs/hardware-dados.md.
"""

import copy
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Ranges:
    friction:   tuple = (0.7, 1.3)    # scale on every geom's sliding friction
    mass:       tuple = (0.9, 1.1)    # scale per body (inertia scaled with it)
    torque:     tuple = (0.85, 1.15)  # scale on actuator force limits
    stiffness:  tuple = (0.85, 1.15)  # scale on position-servo gain kp
    joint_loss: tuple = (0.7, 1.5)    # scale on joint friction loss (gear play)
    latency:    tuple = (0, 2)        # whole control periods of command delay


DEFAULT = Ranges()
NOMINAL = Ranges((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0))


@dataclass(frozen=True)
class Variant:
    model:   mujoco.MjModel
    latency: int        # control periods to delay each new command


def perturb(model: mujoco.MjModel, rng: np.random.Generator,
            r: Ranges = DEFAULT) -> Variant:
    m = copy.copy(model)    # MjModel copy is deep: arrays are duplicated

    m.geom_friction[:, 0] *= rng.uniform(*r.friction)

    mass_scale = rng.uniform(*r.mass, size=m.nbody)
    mass_scale[0] = 1.0                                  # world body
    m.body_mass[:] *= mass_scale
    m.body_inertia[:] *= mass_scale[:, None]

    m.actuator_forcerange[:] *= rng.uniform(*r.torque, size=(m.nu, 1))

    # Position actuators: force = kp * ctrl - kp * qpos, so kp lives in both
    # gainprm[0] and biasprm[1]. Velocity actuators have biasprm[1] == 0.
    is_pos = m.actuator_biasprm[:, 1] != 0
    kp_scale = rng.uniform(*r.stiffness, size=m.nu)
    m.actuator_gainprm[is_pos, 0] *= kp_scale[is_pos]
    m.actuator_biasprm[is_pos, 1] *= kp_scale[is_pos]

    m.dof_frictionloss[:] *= rng.uniform(*r.joint_loss, size=m.nv)

    latency = int(rng.integers(r.latency[0], r.latency[1] + 1))
    return Variant(m, latency)

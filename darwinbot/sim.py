"""
sim.py
------
Episode: one morphology on one terrain, compiled once and replayed for as
many controllers as needed. The inner optimisation loop evaluates hundreds
of controllers per body, so compiling the model only once matters.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from darwinbot.builder import build_mjcf
from darwinbot.controller import CPG
from darwinbot.morphology import Morphology
from darwinbot.terrain import Terrain

EPISODE_SECONDS = 8.0
UPRIGHT_MIN     = 0.5     # torso z-axis . world z below this (60 deg) is a fall


@dataclass
class Metrics:
    dx:       float   # forward displacement of the torso (m)
    dy:       float   # signed lateral displacement (m)
    energy:   float   # mechanical work done by all actuators (J)
    fell:     bool
    unstable: bool    # physics blew up (NaN or runaway velocity)
    duration: float   # simulated seconds before the episode ended
    mass:     float   # total robot mass (kg)


class Episode:
    def __init__(self, morph: Morphology, terrain: Terrain,
                 seconds: float = EPISODE_SECONDS):
        xml, self.layout = build_mjcf(morph, terrain.assets, terrain.world)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)
        self.xml   = xml
        self.morph = morph
        self.terrain = terrain
        self.n_steps = int(seconds / self.model.opt.timestep)
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.mass  = float(self.model.body_subtreemass[self.torso])
        assert self.model.nu == len(self.layout)

    def reset(self, ctrl: np.ndarray) -> CPG:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return CPG(ctrl, self.layout)

    def step(self, cpg: CPG):
        self.data.ctrl[:] = cpg(self.data.time)
        mujoco.mj_step(self.model, self.data)

    def run(self, ctrl: np.ndarray) -> Metrics:
        m, d = self.model, self.data
        cpg  = self.reset(ctrl)
        x0, y0 = d.xpos[self.torso][:2].copy()
        dt = m.opt.timestep
        energy = 0.0
        fell = unstable = False

        for _ in range(self.n_steps):
            self.step(cpg)
            energy += float(np.abs(d.actuator_force * d.actuator_velocity).sum()) * dt
            if not np.isfinite(d.qacc).all() or np.abs(d.qvel).max() > 1e3:
                unstable = True
                break
            if d.xmat[self.torso][8] < UPRIGHT_MIN:
                fell = True
                break

        x, y = d.xpos[self.torso][:2]
        return Metrics(
            dx=float(x - x0) if not unstable else 0.0,
            dy=float(y - y0) if not unstable else 0.0,
            energy=energy, fell=fell, unstable=unstable,
            duration=float(d.time), mass=self.mass,
        )

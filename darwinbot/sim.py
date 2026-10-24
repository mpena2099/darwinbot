"""
sim.py
------
Episode: one morphology on one terrain, compiled once and replayed for as
many controllers as needed. The inner optimisation loop evaluates hundreds
of controllers per body, so compiling the model only once matters.

The controller runs at 50 Hz like the ESP32: servo targets are recomputed
every CONTROL_DT and held in between. A body whose robot geoms already touch
at spawn (overlapping legs) is not buildable and is never simulated.
"""

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from darwinbot.builder import build_mjcf
from darwinbot.controller import CONTROL_DT, CPG
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


def control_hold(timestep: float) -> int:
    """Physics steps per controller update."""
    return max(1, round(CONTROL_DT / timestep))


def is_buildable(model: mujoco.MjModel) -> bool:
    """True when no two robot geoms touch at the spawn pose."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = model.geom_contype == 2
    return not any(robot[c.geom1] and robot[c.geom2] for c in data.contact)


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
        self.hold  = control_hold(self.model.opt.timestep)
        self.buildable = is_buildable(self.model)
        self._k = 0
        assert self.model.nu == len(self.layout)

    def reset(self, ctrl: np.ndarray, model=None) -> CPG:
        model = model or self.model
        mujoco.mj_resetData(model, self.data)
        mujoco.mj_forward(model, self.data)
        self._k = 0
        return CPG(ctrl, self.layout)

    def step(self, cpg: CPG):
        if self._k % self.hold == 0:
            self.data.ctrl[:] = cpg(self.data.time)
        self._k += 1
        mujoco.mj_step(self.model, self.data)

    def run(self, ctrl: np.ndarray, variant=None) -> Metrics:
        """
        Simulate one episode. With a randomize.Variant the perturbed model is
        stepped instead (same sizes, so the MjData is reused) and every new
        command reaches the servos variant.latency control periods late; until
        the first one arrives the targets stay at zero.
        """
        m, d = self.model, self.data
        sim_model = variant.model if variant is not None else m
        cpg  = self.reset(ctrl, sim_model)
        queue = (deque([np.zeros(m.nu)] * variant.latency)
                 if variant is not None and variant.latency > 0 else None)
        x0, y0 = d.xpos[self.torso][:2].copy()
        dt = m.opt.timestep
        energy = 0.0
        fell = unstable = False

        for _ in range(self.n_steps):
            if variant is None:
                self.step(cpg)
            else:
                if self._k % self.hold == 0:
                    target = cpg(d.time)
                    if queue is not None:
                        queue.append(target)
                        target = queue.popleft()
                    d.ctrl[:] = target
                self._k += 1
                mujoco.mj_step(sim_model, d)
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

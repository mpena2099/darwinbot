"""
sim.py
------
Episode: one morphology on one terrain, compiled once and replayed for as
many controllers as needed. The inner optimisation loop evaluates hundreds
of controllers per body, so compiling the model only once matters.

The controller runs at 50 Hz like the ESP32: servo targets are recomputed
every CONTROL_DT and held in between, while the controller's phase state
is integrated every physics step from the actuator forces (load feedback). A body whose robot geoms already touch
at spawn (overlapping legs) is not buildable and is never simulated.
"""

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from darwinbot.builder import build_mjcf
from darwinbot.controller import CONTROL_DT, Controller
from darwinbot.morphology import Morphology, violations
from darwinbot.terrain import Terrain

EPISODE_SECONDS = 8.0
UPRIGHT_MIN     = 0.5     # torso z-axis . world z below this (60 deg) is a fall
HEIGHT_MAX      = 1.5     # clip of the height descriptor (torso z / body reach)
COT_MIN, COT_MAX = 0.1, 100.0   # range of the cost of transport descriptor (log scale)
COT_DX_MIN      = 0.05    # m: a robot that barely moves gets the cost of this distance
TURN_RADIUS     = 0.1     # m: a turn in place counts as an arc of this radius
# MuJoCo resets the simulation on NaN or huge values, so a blow up shows up
# only in these warning counters (and in time jumping back), not in the state.
BLOWUP_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQACC,
                   mujoco.mjtWarning.mjWARN_BADQVEL,
                   mujoco.mjtWarning.mjWARN_BADQPOS)


@dataclass
class Metrics:
    dx:       float   # forward displacement of the torso (m)
    dy:       float   # signed lateral displacement (m)
    energy:   float   # mechanical work done by all actuators (J)
    fell:     bool
    unstable: bool    # physics blew up (NaN or runaway velocity)
    duration: float   # simulated seconds before the episode ended
    mass:     float   # total robot mass (kg)
    duty:     float = 0.0   # mean fraction of control ticks a terminal point touches the ground
    height:   float = 0.0   # mean torso z / body reach, clipped to 0..HEIGHT_MAX
    wheel_share: float = 0.0   # share of terminal contact time that belongs to wheels
    yaw:      float = 0.0   # net change of the torso heading (rad, unwrapped)


def cost_of_transport(m: "Metrics", task: str = "forward") -> float:
    """Mechanical work per weight and distance (no unit), clipped to the
    descriptor range: low is efficient. Robots that go backwards or stay put
    are charged as if they had moved COT_DX_MIN. For the "turn" task the
    distance is the arc swept at TURN_RADIUS. Unstable or non-finite metrics
    give COT_MAX, never NaN."""
    if task == "turn":
        distance = max(abs(m.yaw) * TURN_RADIUS, COT_DX_MIN)
    else:
        distance = max(m.dx, COT_DX_MIN)
    cot = m.energy / (m.mass * 9.81 * distance)
    if m.unstable or not np.isfinite(cot):
        return COT_MAX
    return float(np.clip(cot, COT_MIN, COT_MAX))


def wrap_angle(a):
    """Angle wrapped to [-pi, pi). Works on floats and on JAX arrays."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def heading(xmat):
    """Torso yaw about world Z from a flattened 3x3 rotation matrix."""
    return np.arctan2(xmat[3], xmat[0])


def control_hold(timestep: float) -> int:
    """Physics steps per controller update."""
    return max(1, round(CONTROL_DT / timestep))


def terminal_geoms(model: mujoco.MjModel, morph: Morphology):
    """
    MuJoCo geom id of every terminal point of the body (feet, wheels, free
    tips), in morph.terminal_points() order, and a bool array marking wheels.
    """
    geoms, wheel = [], []
    for i, _, _ in morph.terminal_points():
        is_wheel = morph.nodes[i].kind == "wheel"
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 f"{'w' if is_wheel else 'n'}{i}")
        first = model.body_geomadr[body]
        # physical geoms only: the look-only bracket and horn have contype 0
        own = [g for g in range(first, first + model.body_geomnum[body])
               if model.geom_contype[g] != 0]
        # Links are boxes now; a free tip has no non-box geom to pick.
        non_box = [g for g in own if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX]
        geoms.append(non_box[0] if non_box else
                     max(own, key=lambda g: abs(model.geom_pos[g][0])
                         + model.geom_size[g][0]))
        wheel.append(is_wheel)
    return np.array(geoms, dtype=int), np.array(wheel, dtype=bool)


def behaviour(touch: np.ndarray, wheel: np.ndarray, zsum: float, ticks: int,
              reach: float):
    """(duty, height, wheel_share) from per-terminal touch tick counts."""
    if ticks <= 0:
        return 0.0, 0.0, 0.0
    duty = float(np.mean(touch)) / ticks if len(touch) else 0.0
    total = float(np.sum(touch))
    share = float(np.sum(touch[wheel])) / total if total > 0 else 0.0
    height = float(np.clip(zsum / ticks / reach, 0.0, HEIGHT_MAX))
    return duty, height, share


def is_buildable(model: mujoco.MjModel) -> bool:
    """True when no two robot geoms touch at the spawn pose."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = model.geom_contype == 2
    return not any(robot[c.geom1] and robot[c.geom2] for c in data.contact)


class Episode:
    def __init__(self, morph: Morphology, terrain: Terrain,
                 seconds: float = EPISODE_SECONDS, payload: float = 0.0):
        xml, self.layout = build_mjcf(morph, terrain.assets, terrain.world,
                                      payload=payload,
                                      ground_z=terrain.spawn_z,
                                      terrain_friction=terrain.friction)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)
        self.xml   = xml
        self.morph = morph
        self.terrain = terrain
        self.payload = payload
        self.n_steps = int(seconds / self.model.opt.timestep)
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.mass  = float(self.model.body_subtreemass[self.torso])
        self.hold  = control_hold(self.model.opt.timestep)
        self.buildable = is_buildable(self.model) and not violations(morph)
        self.reach = morph.max_reach
        self._robot = self.model.geom_contype == 2
        self._tgeoms, self._twheel = terminal_geoms(self.model, morph)
        self._k = 0
        assert self.model.nu == len(self.layout)

    def reset(self, ctrl: np.ndarray, model=None, command: float = 0.0) -> Controller:
        model = model or self.model
        mujoco.mj_resetData(model, self.data)
        mujoco.mj_forward(model, self.data)
        self._k = 0
        for w in BLOWUP_WARNINGS:
            self.data.warning[w].number = 0
        return Controller(ctrl, self.layout, command)

    def _blew_up(self, t_prev: float) -> bool:
        """True when the last mj_step diverged: a MuJoCo warning fired (it has
        already reset the state), time went backwards, or the state is bad."""
        d = self.data
        return (any(d.warning[w].number > 0 for w in BLOWUP_WARNINGS)
                or d.time < t_prev
                or not np.isfinite(d.qacc).all()
                or bool(np.abs(d.qvel).max() > 1e3))

    def step(self, controller: Controller):
        if self._k % self.hold == 0:
            self.data.ctrl[:] = controller.targets(self.data.time)
        self._k += 1
        mujoco.mj_step(self.model, self.data)
        controller.advance(self.model.opt.timestep, self.data.actuator_force)

    def _touching(self) -> np.ndarray:
        """Per terminal point: any contact with a non-robot geom right now."""
        d = self.data
        g = d.contact.geom[:d.ncon]
        a, b = g[:, 0], g[:, 1]
        hit = np.concatenate([a[~self._robot[b]], b[~self._robot[a]]])
        return np.isin(self._tgeoms, hit)

    def run(self, ctrl: np.ndarray, variant=None, command: float = 0.0) -> Metrics:
        """
        Simulate one episode. With a randomize.Variant the perturbed model is
        stepped instead (same sizes, so the MjData is reused) and every new
        command reaches the servos variant.latency control periods late; until
        the first one arrives the targets stay at zero.
        """
        m, d = self.model, self.data
        sim_model = variant.model if variant is not None else m
        controller = self.reset(ctrl, sim_model, command)
        queue = (deque([np.zeros(m.nu)] * variant.latency)
                 if variant is not None and variant.latency > 0 else None)
        x0, y0 = d.xpos[self.torso][:2].copy()
        dt = m.opt.timestep
        energy = 0.0
        yaw, prev = 0.0, heading(d.xmat[self.torso])
        fell = unstable = False
        t_prev = 0.0
        touch = np.zeros(len(self._tgeoms))
        zsum, ticks = 0.0, 0

        for n in range(1, self.n_steps + 1):
            if variant is None:
                self.step(controller)
            else:
                if self._k % self.hold == 0:
                    target = controller.targets(d.time)
                    if queue is not None:
                        queue.append(target)
                        target = queue.popleft()
                    d.ctrl[:] = target
                self._k += 1
                mujoco.mj_step(sim_model, d)
                controller.advance(dt, d.actuator_force)
            energy += float(np.abs(d.actuator_force * d.actuator_velocity).sum()) * dt
            # Per-step wrapped differences: the sum is the unwrapped heading change.
            now = heading(d.xmat[self.torso])
            yaw += float(wrap_angle(now - prev))
            prev = now
            if self._blew_up(t_prev):
                unstable = True
                break
            t_prev = d.time
            if d.xmat[self.torso][8] < UPRIGHT_MIN:
                fell = True
                break
            if n % self.hold == 0:          # behaviour sampled once per control tick
                touch += self._touching()
                zsum += float(d.xpos[self.torso][2])
                ticks += 1

        x, y = d.xpos[self.torso][:2]
        duty, height, share = behaviour(touch, self._twheel, zsum, ticks, self.reach)
        return Metrics(
            dx=float(x - x0) if not unstable else 0.0,
            dy=float(y - y0) if not unstable else 0.0,
            energy=energy, fell=fell, unstable=unstable,
            duration=float(d.time), mass=self.mass,
            duty=duty, height=height, wheel_share=share,
            yaw=yaw if not unstable else 0.0,
        )

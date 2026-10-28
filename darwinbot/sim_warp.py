"""
sim_warp.py
-----------
WarpBatchEpisode: the same body, terrain and controller semantics as
sim.Episode, simulated with MuJoCo Warp. Every world in the batch is one
controller on one copy of the body, so a CMA-ES population is a single batch.

One control period (servo targets, `hold` physics steps with a phase and
book-keeping kernel after each of them, then the behaviour sample) is the unit
of work, and on CUDA it is recorded once as a graph and replayed. A done world
keeps stepping but its metrics freeze at the step it fell (or blew up), so the
metrics mean the same as sim.Episode. A non-buildable body is reported the way
inner.py reports it.
"""

from typing import List

import mujoco_warp as mjw
import numpy as np
import warp as wp

from darwinbot.controller import CTRL_LENGTH, RAMP_TIME, SERVO_TORQUE, Params
from darwinbot.morphology import Morphology
from darwinbot.sim import (EPISODE_SECONDS, UPRIGHT_MIN, Metrics, Episode,
                           behaviour)
from darwinbot.terrain import Terrain

NCONMAX = 96        # contacts per world; the default guess overflows on a 4-legged body
NJMAX = 256         # constraint rows per world: 4 per contact, plus limits
RUNAWAY_VEL = 1e3   # same threshold as sim._bad_state


@wp.kernel
def _targets(
    time: wp.array(dtype=float),
    theta: wp.array2d(dtype=float),
    amp1: wp.array2d(dtype=float),
    phase1: wp.array2d(dtype=float),
    offset: wp.array2d(dtype=float),
    amp2: wp.array2d(dtype=float),
    phase2: wp.array2d(dtype=float),
    ramp_time: float,
    ctrl: wp.array2d(dtype=float),
):
    """Servo targets from the current phases, as controller.servo_targets."""
    w, a = wp.tid()
    th = theta[w, a]
    fade = wp.min(1.0, time[w] / ramp_time)
    ctrl[w, a] = fade * (offset[w, a] + amp1[w, a] * wp.sin(th + phase1[w, a])
                         + amp2[w, a] * wp.sin(2.0 * th + phase2[w, a]))


@wp.kernel
def _advance_phases(
    theta: wp.array2d(dtype=float),
    omega: wp.array(dtype=float),
    sigma: wp.array2d(dtype=float),
    force: wp.array2d(dtype=float),
    nu: int,
    dt: float,
):
    """One Euler step of the phases, as controller.advance_phases."""
    w = wp.tid()
    for a in range(nu):
        load = force[w, a] / SERVO_TORQUE
        if load > 1.0:
            load = 1.0
        elif load < -1.0:
            load = -1.0
        th = theta[w, a]
        theta[w, a] = th + dt * (omega[w] - sigma[w, a] * load * wp.cos(th))


@wp.kernel
def _book_keep(
    time: wp.array(dtype=float),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    force: wp.array2d(dtype=float),
    velocity: wp.array2d(dtype=float),
    qacc: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    torso: int,
    nu: int,
    nv: int,
    dt: float,
    x0: wp.array(dtype=float),
    y0: wp.array(dtype=float),
    done: wp.array(dtype=int),
    dx: wp.array(dtype=float),
    dy: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    duration: wp.array(dtype=float),
    yaw: wp.array(dtype=float),
    prev: wp.array(dtype=float),
    fell: wp.array(dtype=int),
    unstable: wp.array(dtype=int),
):
    """Work, heading, displacement and the end checks of one step, as in
    sim.Episode.run: unstable wins over fell, and a done world is frozen."""
    w = wp.tid()
    if done[w] != 0:
        return
    work = float(0.0)
    for a in range(nu):
        work += wp.abs(force[w, a] * velocity[w, a])
    energy[w] = energy[w] + work * dt
    rot = xmat[w, torso]
    now = wp.atan2(rot[1, 0], rot[0, 0])
    # Wrapped per-step differences add up to the unwrapped heading change.
    dif = now - prev[w]
    yaw[w] = yaw[w] + dif - 2.0 * wp.pi * wp.floor((dif + wp.pi) / (2.0 * wp.pi))
    prev[w] = now
    pos = xpos[w, torso]
    dx[w] = pos[0] - x0[w]
    dy[w] = pos[1] - y0[w]
    duration[w] = time[w]
    bad = int(0)
    for i in range(nv):
        if not wp.isfinite(qacc[w, i]) or wp.abs(qvel[w, i]) > RUNAWAY_VEL:
            bad = 1
    if bad != 0:
        unstable[w] = 1
        done[w] = 1
    elif rot[2, 2] < UPRIGHT_MIN:
        fell[w] = 1
        done[w] = 1


@wp.kernel
def _sample(
    xpos: wp.array2d(dtype=wp.vec3),
    torso: int,
    done: wp.array(dtype=int),
    zsum: wp.array(dtype=float),
    ticks: wp.array(dtype=int),
):
    """Torso height of one behaviour tick."""
    w = wp.tid()
    if done[w] != 0:
        return
    zsum[w] = zsum[w] + xpos[w, torso][2]
    ticks[w] = ticks[w] + 1


@wp.kernel
def _contacts(
    nacon: wp.array(dtype=int),
    worldid: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    robot: wp.array(dtype=int),
    tip_of: wp.array(dtype=int),
    hit: wp.array2d(dtype=float),
):
    """One thread per contact slot: mark the terminal point it touches, if the
    other geom is not part of the robot (as Episode._touching)."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    w = worldid[c]
    a = geom[c][0]
    b = geom[c][1]
    if tip_of[a] >= 0 and robot[b] == 0:
        hit[w, tip_of[a]] = 1.0
    if tip_of[b] >= 0 and robot[a] == 0:
        hit[w, tip_of[b]] = 1.0


@wp.kernel
def _touch(
    done: wp.array(dtype=int),
    hit: wp.array2d(dtype=float),
    touch: wp.array2d(dtype=float),
):
    """Add this tick's touches of the worlds still running, then clear them."""
    w, k = wp.tid()
    if done[w] == 0:
        touch[w, k] = touch[w, k] + hit[w, k]
    hit[w, k] = 0.0


class WarpBatchEpisode:
    """One body on one terrain, replayed for a batch of controllers at once."""

    def __init__(self, morph: Morphology, terrain: Terrain,
                 seconds: float = EPISODE_SECONDS, payload: float = 0.0,
                 device=None):
        if device is not None:
            wp.set_device(device)
        self.device = wp.get_device()
        self.episode = Episode(morph, terrain, seconds, payload)
        ep = self.episode
        self.layout = ep.layout
        self.buildable = ep.buildable
        self.mass = ep.mass
        self.hold = ep.hold
        self.n_steps = ep.n_steps
        self.dt = float(ep.model.opt.timestep)
        self.torso = ep.torso
        self.reach = ep.reach
        self.twheel = ep._twheel
        self._cache = {}
        if not self.buildable:
            return                  # run_batch reports it, as inner._unbuildable
        self.model = mjw.put_model(ep.model)
        self.nu, self.nv = ep.model.nu, ep.model.nv
        dev = self.device
        self.n_tips = len(ep._tgeoms)
        tip_of = np.full(ep.model.ngeom, -1, dtype=np.int32)   # geom -> terminal index
        tip_of[np.asarray(ep._tgeoms)] = np.arange(self.n_tips, dtype=np.int32)
        self.tip_of = wp.array(tip_of, dtype=wp.int32, device=dev)
        self.robot = wp.array(np.asarray(ep._robot, dtype=np.int32),
                              dtype=wp.int32, device=dev)

    # ---- device buffers, one set per batch size ----

    def _batch(self, n: int) -> dict:
        b = self._cache.get(n)
        if b is not None:
            return b
        dev = self.device
        nu = self.nu

        def zeros(shape, dtype=wp.float32):
            return wp.zeros(shape, dtype=dtype, device=dev)

        b = {"d": mjw.make_data(self.episode.model, nworld=n,
                                nconmax=NCONMAX, njmax=NJMAX),
             "omega": zeros(n), "x0": zeros(n), "y0": zeros(n),
             "prev": zeros(n), "dx": zeros(n), "dy": zeros(n),
             "energy": zeros(n), "duration": zeros(n), "yaw": zeros(n),
             "zsum": zeros(n), "ticks": zeros(n, wp.int32),
             "done": zeros(n, wp.int32), "fell": zeros(n, wp.int32),
             "unstable": zeros(n, wp.int32), "touch": zeros((n, self.n_tips)), "hit": zeros((n, self.n_tips)),
             "theta": zeros((n, nu))}
        for name in ("amp1", "phase1", "offset", "amp2", "phase2", "sigma"):
            b[name] = zeros((n, nu))
        b["graph"] = self._capture(b) if self.device.is_cuda else None
        self._cache[n] = b
        return b

    def _capture(self, b: dict):
        """Record one control period, for replay with wp.capture_launch."""
        with wp.ScopedCapture() as capture:
            self._period(b, self.hold, sample=True)
        return capture.graph

    # ---- the loop ----

    def _period(self, b: dict, steps: int, sample: bool) -> None:
        """Servo targets, `steps` physics steps, then one behaviour sample."""
        d = b["d"]
        dev = self.device
        wp.launch(_targets, dim=(d.nworld, self.nu), device=dev, inputs=[
            d.time, b["theta"], b["amp1"], b["phase1"], b["offset"], b["amp2"],
            b["phase2"], RAMP_TIME], outputs=[d.ctrl])
        for _ in range(steps):
            mjw.step(self.model, d)
            wp.launch(_advance_phases, dim=d.nworld, device=dev, inputs=[
                b["theta"], b["omega"], b["sigma"], d.actuator_force,
                self.nu, self.dt])
            wp.launch(_book_keep, dim=d.nworld, device=dev, inputs=[
                d.time, d.xpos, d.xmat, d.actuator_force, d.actuator_velocity,
                d.qacc, d.qvel, self.torso, self.nu, self.nv, self.dt,
                b["x0"], b["y0"], b["done"]],
                outputs=[b["dx"], b["dy"], b["energy"], b["duration"],
                         b["yaw"], b["prev"], b["fell"], b["unstable"]])
        if not sample:
            return
        wp.launch(_sample, dim=d.nworld, device=dev, inputs=[
            d.xpos, self.torso, b["done"]], outputs=[b["zsum"], b["ticks"]])
        wp.launch(_contacts, dim=d.contact.geom.shape[0], device=dev,
                  inputs=[d.nacon, d.contact.worldid, d.contact.geom,
                          self.robot, self.tip_of], outputs=[b["hit"]])
        wp.launch(_touch, dim=(d.nworld, self.n_tips), device=dev,
                  inputs=[b["done"]], outputs=[b["hit"], b["touch"]])

    def _reset(self, b: dict) -> None:
        """Every world back to the spawn pose (mj_resetData + mj_forward)."""
        d = b["d"]
        mjw.reset_data(self.model, d)
        mjw.forward(self.model, d)
        for name, arr in b.items():
            if name not in ("d", "graph"):
                arr.zero_()
        # The spawn pose is the same in every world: read it once.
        xpos = d.xpos.numpy()[:, self.torso]
        xmat = d.xmat.numpy()[:, self.torso]
        b["x0"].assign(xpos[:, 0])
        b["y0"].assign(xpos[:, 1])
        b["prev"].assign(np.arctan2(xmat[:, 1, 0], xmat[:, 0, 0]))

    def run_batch(self, ctrls: np.ndarray, commands=0.0) -> List[Metrics]:
        ctrls = np.asarray(ctrls, dtype=float).reshape(-1, CTRL_LENGTH)
        n = len(ctrls)
        assert n > 0
        if not self.buildable:
            return [Metrics(0.0, 0.0, 0.0, False, True, 0.0, self.mass)] * n
        cmds = np.broadcast_to(np.asarray(commands, dtype=float), (n,))
        ps = [Params(c, self.layout, float(cmd)) for c, cmd in zip(ctrls, cmds)]
        b = self._batch(n)
        self._reset(b)
        b["omega"].assign(np.array([p.omega for p in ps]))
        for name in ("amp1", "phase1", "offset", "amp2", "phase2", "sigma"):
            b[name].assign(np.stack([getattr(p, name) for p in ps]))
        periods, rest = divmod(self.n_steps, self.hold)
        for _ in range(periods):
            if b["graph"] is not None:
                wp.capture_launch(b["graph"])
            else:
                self._period(b, self.hold, sample=True)
        if rest:                       # a last partial period samples nothing
            self._period(b, rest, sample=False)
        return self._metrics(b, n)

    def _metrics(self, b: dict, n: int) -> List[Metrics]:
        """Copies back and builds Metrics as sim.Episode.run does."""
        out = {k: b[k].numpy() for k in ("dx", "dy", "energy", "duration", "yaw",
                                         "fell", "unstable", "zsum", "ticks",
                                         "touch")}
        out["touch"] = out["touch"].astype(float)
        res = []
        for i in range(n):
            duty, height, share = behaviour(out["touch"][i], self.twheel,
                                            float(out["zsum"][i]),
                                            int(out["ticks"][i]), self.reach)
            unstable = bool(out["unstable"][i])
            # Same convention as Episode.run: a blow up reports no displacement.
            keep = 0.0 if unstable else 1.0
            res.append(Metrics(
                dx=float(out["dx"][i]) * keep, dy=float(out["dy"][i]) * keep,
                energy=float(out["energy"][i]), fell=bool(out["fell"][i]),
                unstable=unstable, duration=float(out["duration"][i]),
                mass=self.mass, duty=duty, height=height, wheel_share=share,
                yaw=float(out["yaw"][i]) * keep))
        return res

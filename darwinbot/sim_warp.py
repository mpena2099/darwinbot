"""
sim_warp.py
-----------
WarpBatchEpisode: the same body, terrain and controller semantics as
sim.Episode, simulated with MuJoCo Warp. Every world in the batch is one
controller on one copy of the body, with its own physics and command latency
when given a randomize.Variant, so a CMA-ES population is a single batch.

One control period (servo targets, `hold` physics steps with a phase and
book-keeping kernel after each of them, then the behaviour sample) is the unit
of work, and on CUDA it is recorded once as a graph and replayed. A done world
keeps stepping but its metrics freeze at the step it fell (or blew up), so the
metrics mean the same as sim.Episode. A non-buildable body is reported the way
inner.py reports it.
"""

from contextlib import nullcontext
from typing import List

import mujoco

import mujoco_warp as mjw
import numpy as np
import warp as wp

from darwinbot.controller import CTRL_LENGTH, RAMP_TIME, SERVO_TORQUE, Params
from darwinbot.morphology import Morphology
from darwinbot.sim import (EPISODE_SECONDS, UPRIGHT_MIN, Metrics, Episode,
                           behaviour)
from darwinbot.terrain import Terrain

NCONMAX = 128       # contacts per world; the default guess overflows on a 4-legged body
NJMAX = 1024        # constraint rows per world: 256 overflowed on large evolved bodies
PAD = 32            # batch sizes round up to this, so few graphs are captured per body
RUNAWAY_VEL = 1e3   # same threshold as sim._bad_state
VARIANT_FIELDS = ("geom_friction", "body_mass", "body_inertia",
                  "actuator_forcerange", "actuator_gainprm", "actuator_biasprm",
                  "dof_frictionloss", "dof_damping")   # exactly what randomize.perturb changes


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
    targets: wp.array2d(dtype=float),
):
    """Servo targets from the current phases, as controller.servo_targets."""
    w, a = wp.tid()
    th = theta[w, a]
    fade = wp.min(1.0, time[w] / ramp_time)
    targets[w, a] = fade * (offset[w, a] + amp1[w, a] * wp.sin(th + phase1[w, a])
                            + amp2[w, a] * wp.sin(2.0 * th + phase2[w, a]))


@wp.kernel
def _delay(
    targets: wp.array2d(dtype=float),
    latency: wp.array(dtype=int),
    ring: wp.array3d(dtype=float),
    tick: wp.array(dtype=int),
    nu: int,
    nlat: int,
    ctrl: wp.array2d(dtype=float),
):
    """Push this tick's targets through the per-world ring, as the deque of
    sim.Episode.run: a target reaches the servos `latency` ticks late and the
    first ones are zero. The counter lives on the device so a graph replays it.
    """
    w = wp.tid()
    t = tick[w]
    for a in range(nu):
        ring[w, t % nlat, a] = targets[w, a]
        ctrl[w, a] = ring[w, (t - latency[w] + nlat) % nlat, a]
    tick[w] = t + 1


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
    mid: int,
    dx: wp.array(dtype=float),
    dy: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    duration: wp.array(dtype=float),
    yaw: wp.array(dtype=float),
    prev: wp.array(dtype=float),
    fell: wp.array(dtype=int),
    unstable: wp.array(dtype=int),
    step: wp.array(dtype=int),
    x_mid: wp.array(dtype=float),
    reached: wp.array(dtype=int),
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
    # Half-way mark, as in Episode.run: only if the episode is still running.
    step[w] = step[w] + 1
    if step[w] == mid and done[w] == 0:
        x_mid[w] = dx[w]
        reached[w] = 1


@wp.kernel
def _sample(
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    torso: int,
    done: wp.array(dtype=int),
    zsum: wp.array(dtype=float),
    upsum: wp.array(dtype=float),
    ticks: wp.array(dtype=int),
):
    """Torso height and uprightness of one behaviour tick."""
    w = wp.tid()
    if done[w] != 0:
        return
    zsum[w] = zsum[w] + xpos[w, torso][2]
    upsum[w] = upsum[w] + xmat[w, torso][2, 2]
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
        self._vmodels = {}
        if not self.buildable:
            return                  # run_batch reports it, as inner._unbuildable
        self.model = mjw.put_model(ep.model)
        self.model.opt.warn_overflow = 0      # overflowed worlds are marked unstable instead
        # Every world starts at the same spawn pose: torso x, y and heading.
        mujoco.mj_resetData(ep.model, ep.data)
        mujoco.mj_forward(ep.model, ep.data)
        pos, mat = ep.data.xpos[ep.torso], ep.data.xmat[ep.torso]
        self._spawn = (float(pos[0]), float(pos[1]), float(np.arctan2(mat[3], mat[0])))
        # Its own CUDA stream, so the terrains of one generation run concurrently.
        self.stream = wp.Stream(self.device) if self.device.is_cuda else None
        self.nu, self.nv = ep.model.nu, ep.model.nv
        dev = self.device
        self.n_tips = len(ep._tgeoms)
        tip_of = np.full(ep.model.ngeom, -1, dtype=np.int32)   # geom -> terminal index
        tip_of[np.asarray(ep._tgeoms)] = np.arange(self.n_tips, dtype=np.int32)
        self.tip_of = wp.array(tip_of, dtype=wp.int32, device=dev)
        self.robot = wp.array(np.asarray(ep._robot, dtype=np.int32),
                              dtype=wp.int32, device=dev)

    # ---- device buffers, one set per batch size ----

    def _batch_model(self, n: int):
        """The model with the fields a variant can change, in n copies."""
        m = self._vmodels.get(n)
        if m is None:
            m = mjw.put_model(self.episode.model,
                              batch_sizes={f: n for f in VARIANT_FIELDS})
            m.opt.warn_overflow = 0
            self._vmodels[n] = m
        return m

    def _batch(self, n: int, variants=None) -> dict:
        """Buffers for n worlds. A variant batch carries the model fields per
        world and a ring as deep as its deepest delay, so it is cached apart."""
        nlat = 1 + max([v.latency for v in variants or () if v is not None] or [0])
        key = n if variants is None else (n, nlat)
        b = self._cache.get(key)
        if b is not None:
            return b
        dev = self.device
        nu = self.nu
        model = self.model if variants is None else self._batch_model(n)

        def zeros(shape, dtype=wp.float32):
            return wp.zeros(shape, dtype=dtype, device=dev)

        b = {"model": model,
             "d": mjw.make_data(self.episode.model, nworld=n,
                                 nconmax=NCONMAX, njmax=NJMAX),
             "omega": zeros(n), "x0": zeros(n), "y0": zeros(n),
             "prev": zeros(n), "dx": zeros(n), "dy": zeros(n),
             "energy": zeros(n), "duration": zeros(n), "yaw": zeros(n),
             "step": zeros(n, wp.int32), "x_mid": zeros(n), "reached": zeros(n, wp.int32),
             "zsum": zeros(n), "upsum": zeros(n), "ticks": zeros(n, wp.int32),
             "done": zeros(n, wp.int32), "fell": zeros(n, wp.int32),
             "unstable": zeros(n, wp.int32), "touch": zeros((n, self.n_tips)), "hit": zeros((n, self.n_tips)),
             "theta": zeros((n, nu)), "target": zeros((n, nu)),
             "ring": zeros((n, nlat, nu)),
             "latency": zeros(n, wp.int32), "tick": zeros(n, wp.int32)}
        for name in ("amp1", "phase1", "offset", "amp2", "phase2", "sigma"):
            b[name] = zeros((n, nu))
        b["graph"] = self._capture(b) if self.device.is_cuda else None
        self._cache[key] = b
        return b

    def _capture(self, b: dict):
        """Record one control period, for replay with wp.capture_launch."""
        with self._on_stream(), wp.ScopedCapture() as capture:
            self._period(b, self.hold, sample=True)
        return capture.graph

    def _on_stream(self):
        return wp.ScopedStream(self.stream) if self.stream is not None else nullcontext()

    # ---- the loop ----

    def _period(self, b: dict, steps: int, sample: bool) -> None:
        """Servo targets, `steps` physics steps, then one behaviour sample."""
        d = b["d"]
        dev = self.device
        wp.launch(_targets, dim=(d.nworld, self.nu), device=dev, inputs=[
            d.time, b["theta"], b["amp1"], b["phase1"], b["offset"], b["amp2"],
            b["phase2"], RAMP_TIME], outputs=[b["target"]])
        wp.launch(_delay, dim=d.nworld, device=dev, inputs=[
            b["target"], b["latency"], b["ring"], b["tick"], self.nu,
            b["ring"].shape[1]], outputs=[d.ctrl])
        for _ in range(steps):
            mjw.step(b["model"], d)
            wp.launch(_advance_phases, dim=d.nworld, device=dev, inputs=[
                b["theta"], b["omega"], b["sigma"], d.actuator_force,
                self.nu, self.dt])
            wp.launch(_book_keep, dim=d.nworld, device=dev, inputs=[
                d.time, d.xpos, d.xmat, d.actuator_force, d.actuator_velocity,
                d.qacc, d.qvel, self.torso, self.nu, self.nv, self.dt,
                b["x0"], b["y0"], b["done"], self.n_steps // 2],
                outputs=[b["dx"], b["dy"], b["energy"], b["duration"],
                         b["yaw"], b["prev"], b["fell"], b["unstable"],
                         b["step"], b["x_mid"], b["reached"]])
        if not sample:
            return
        wp.launch(_sample, dim=d.nworld, device=dev, inputs=[
            d.xpos, d.xmat, self.torso, b["done"]],
            outputs=[b["zsum"], b["upsum"], b["ticks"]])
        wp.launch(_contacts, dim=d.contact.geom.shape[0], device=dev,
                  inputs=[d.nacon, d.contact.worldid, d.contact.geom,
                          self.robot, self.tip_of], outputs=[b["hit"]])
        wp.launch(_touch, dim=(d.nworld, self.n_tips), device=dev,
                  inputs=[b["done"]], outputs=[b["hit"], b["touch"]])

    def _reset(self, b: dict) -> None:
        """Every world back to the spawn pose (mj_resetData + mj_forward)."""
        d = b["d"]
        mjw.reset_data(b["model"], d)
        mjw.forward(b["model"], d)
        for arr in b.values():
            if isinstance(arr, wp.array):
                arr.zero_()
        b["x0"].fill_(self._spawn[0])
        b["y0"].fill_(self._spawn[1])
        b["prev"].fill_(self._spawn[2])

    def _variant_fields(self, model, variants: list) -> None:
        """The fields randomize.perturb changes, one world per variant."""
        nominal = self.episode.model
        for f in VARIANT_FIELDS:
            getattr(model, f).assign(np.stack([
                getattr(v.model if v is not None else nominal, f)
                for v in variants]))

    def run_batch(self, ctrls: np.ndarray, commands=0.0,
                  variants=None) -> List[Metrics]:
        """One episode per controller. `variants` is one randomize.Variant per
        world (None is the nominal one): its physics replaces the model fields
        and its latency delays the servo targets, as sim.Episode.run."""
        return self.finish(self.start(ctrls, commands, variants))

    def start(self, ctrls: np.ndarray, commands=0.0, variants=None):
        """Launch run_batch without waiting for it (on this episode's stream);
        finish() collects the metrics. Batches are padded to a multiple of PAD
        with copies of the first world."""
        ctrls = np.asarray(ctrls, dtype=float).reshape(-1, CTRL_LENGTH)
        n = len(ctrls)
        assert n > 0
        if variants is not None:
            assert len(variants) == n
        if not self.buildable:
            return n, None
        cmds = np.broadcast_to(np.asarray(commands, dtype=float), (n,))
        size = -(-n // PAD) * PAD
        pad = [0] * (size - n)
        ctrls = np.concatenate([ctrls, ctrls[pad]])
        cmds = np.concatenate([cmds, cmds[pad]])
        if variants is not None:
            variants = list(variants) + [variants[0]] * len(pad)
        ps = [Params(c, self.layout, float(cmd)) for c, cmd in zip(ctrls, cmds)]
        b = self._batch(size, variants)
        with self._on_stream():
            self._reset(b)
            if variants is not None:
                self._variant_fields(b["model"], variants)
                b["latency"].assign(np.array(
                    [v.latency if v is not None else 0 for v in variants],
                    dtype=np.int32))
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
        return n, b

    def finish(self, token) -> List[Metrics]:
        n, b = token
        if b is None:
            return [Metrics(0.0, 0.0, 0.0, False, True, 0.0, self.mass)] * n
        if self.stream is not None:
            wp.synchronize_stream(self.stream)
        return self._metrics(b, n)

    def _metrics(self, b: dict, n: int) -> List[Metrics]:
        """Copies back and builds Metrics as sim.Episode.run does."""
        out = {k: b[k].numpy() for k in ("dx", "dy", "energy", "duration", "yaw",
                                         "fell", "unstable", "zsum", "upsum", "ticks",
                                         "touch", "x_mid", "reached")}
        out["touch"] = out["touch"].astype(float)
        # A world that ran out of contact or constraint room was simulated wrong.
        out["unstable"] = out["unstable"] | (b["d"].overflow.numpy() != 0)
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
                # Same as Episode.run: 2 * (x_end - x_mid) once half-way was reached.
                progress=(float(out["dx"][i]) if not out["reached"][i]
                          else 2.0 * float(out["dx"][i] - out["x_mid"][i])) * keep,
                energy=float(out["energy"][i]), fell=bool(out["fell"][i]),
                unstable=unstable, duration=float(out["duration"][i]),
                mass=self.mass, duty=duty, height=height, wheel_share=share,
                yaw=float(out["yaw"][i]) * keep,
                upright=(float(out["upsum"][i]) / int(out["ticks"][i])
                         if out["ticks"][i] else 1.0)))
        return res

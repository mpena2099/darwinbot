"""
sim_gpu.py
----------
BatchEpisode: the same body and terrain as sim.Episode, simulated with MJX on
the GPU. The whole CMA-ES population runs as one vmap batch, so one compiled
model evaluates hundreds of controllers at once.

The scan never stops early: a done flag freezes dx, dy, energy and duration at
the step the robot fell (or the physics blew up), so the metrics mean the same
as sim.Episode. Servo targets update at 50 Hz and are held in between, as
there. A non-buildable body skips the MJX compile. JAX and MJX are imported here only; the CPU path does not need
the optional extra.
"""

from typing import List

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from darwinbot.builder import build_mjcf
from darwinbot.controller import CPG, CTRL_LENGTH, RAMP_TIME
from darwinbot.morphology import Morphology
from darwinbot.sim import (EPISODE_SECONDS, UPRIGHT_MIN, Metrics,
                           control_hold, is_buildable)
from darwinbot.terrain import Terrain

BATCH_ALIGN = 8          # pad small batches so x0 reuses the compiled shape


class BatchEpisode:
    def __init__(self, morph: Morphology, terrain: Terrain,
                 seconds: float = EPISODE_SECONDS):
        xml, self.layout = build_mjcf(morph, terrain.assets, terrain.world)
        model = mujoco.MjModel.from_xml_string(xml)
        self.dt = float(model.opt.timestep)
        self.buildable = is_buildable(model)
        self.mass = float(model.body_subtreemass[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")])
        if not self.buildable:
            return
        self.model = mjx.put_model(model)
        self.hold = control_hold(self.dt)
        self.n_steps = int(seconds / self.dt)
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.d0 = mjx.forward(self.model, mjx.make_data(self.model))
        self._run = jax.jit(jax.vmap(self._episode))

    def _episode(self, omega, amp, offset, phase):
        mx, torso, dt = self.model, self.torso, self.dt
        x0 = self.d0.xpos[torso, 0]
        y0 = self.d0.xpos[torso, 1]

        def body(carry, k):
            d, held, dx, dy, energy, duration, done, unstable = carry
            t = d.time
            ramp = jnp.minimum(1.0, t / RAMP_TIME)
            new = ramp * (offset + amp * jnp.sin(omega * t + phase))
            held = jnp.where(k % self.hold == 0, new, held)   # 50 Hz hold
            d = mjx.step(mx, d.replace(ctrl=held))
            work = jnp.abs(d.actuator_force * d._impl.actuator_velocity).sum() * dt
            blew_up = ~jnp.isfinite(d.qacc).all() | (jnp.abs(d.qvel).max() > 1e3)
            ended = blew_up | (d.xmat[torso, 2, 2] < UPRIGHT_MIN)
            first = ended & ~done
            dx = jnp.where(done, dx, d.xpos[torso, 0] - x0)
            dy = jnp.where(done, dy, d.xpos[torso, 1] - y0)
            energy = jnp.where(done, energy, energy + work)
            duration = jnp.where(done, duration, d.time)
            unstable = jnp.where(first, blew_up, unstable)
            return (d, held, dx, dy, energy, duration, done | ended, unstable), None

        carry = (self.d0, jnp.zeros_like(amp), 0.0, 0.0, 0.0, 0.0, False, False)
        (d, _, dx, dy, energy, duration, done, unstable), _ = jax.lax.scan(
            body, carry, jnp.arange(self.n_steps))
        # Same convention as Episode: a blow up reports no displacement.
        keep = jnp.where(unstable, 0.0, 1.0)
        # As in Episode, a blow up is reported as unstable, not as a fall.
        return dx * keep, dy * keep, energy, done & ~unstable, unstable, duration

    def run_batch(self, ctrls: np.ndarray) -> List[Metrics]:
        ctrls = np.asarray(ctrls, dtype=float).reshape(-1, CTRL_LENGTH)
        n = len(ctrls)
        assert n > 0
        pad = -n % BATCH_ALIGN
        if pad:
            ctrls = np.concatenate([ctrls, np.repeat(ctrls[-1:], pad, axis=0)])
        cpg = [CPG(c, self.layout) for c in ctrls]
        out = self._run(
            jnp.asarray([c.omega for c in cpg]),
            jnp.asarray(np.stack([c.amp for c in cpg])),
            jnp.asarray(np.stack([c.offset for c in cpg])),
            jnp.asarray(np.stack([c.phase for c in cpg])),
        )
        dx, dy, energy, fell, unstable, duration = (np.asarray(v) for v in out)
        return [
            Metrics(dx=float(dx[i]), dy=float(dy[i]), energy=float(energy[i]),
                    fell=bool(fell[i]), unstable=bool(unstable[i]),
                    duration=float(duration[i]), mass=self.mass)
            for i in range(n)
        ]

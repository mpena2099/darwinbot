"""
sim_gpu.py
----------
BatchEpisode: the same body and terrain as sim.Episode, simulated with MJX on
the GPU. The whole CMA-ES population runs as one vmap batch, so one compiled
model evaluates hundreds of controllers at once.

The scan never stops early: a done flag freezes dx, dy, energy and duration at
the step the robot fell (or the physics blew up), so the metrics mean the same
as sim.Episode. Servo targets update at 50 Hz and are held in between, as
there. The per-joint phases ride in the scan carry and are integrated every
physics step from the actuator forces, as in sim.Episode. A non-buildable body skips the MJX compile. JAX and MJX are imported here only; the CPU path does not need
the optional extra.
"""

from typing import List

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from darwinbot.builder import build_mjcf
from darwinbot.controller import (CTRL_LENGTH, RAMP_TIME, SERVO_TORQUE, Params)
from darwinbot.morphology import Morphology, violations
from darwinbot.sim import (EPISODE_SECONDS, UPRIGHT_MIN, Metrics, behaviour,
                           control_hold, is_buildable, terminal_geoms,
                           wrap_angle)
from darwinbot.terrain import Terrain

BATCH_ALIGN = 8          # pad small batches so x0 reuses the compiled shape


class BatchEpisode:
    def __init__(self, morph: Morphology, terrain: Terrain,
                 seconds: float = EPISODE_SECONDS, payload: float = 0.0):
        xml, self.layout = build_mjcf(morph, terrain.assets, terrain.world,
                                      payload=payload,
                                      ground_z=terrain.spawn_z,
                                      terrain_friction=terrain.friction)
        model = mujoco.MjModel.from_xml_string(xml)
        self.dt = float(model.opt.timestep)
        self.buildable = is_buildable(model, morph) and not violations(morph)
        self.mass = float(model.body_subtreemass[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")])
        if not self.buildable:
            return
        self.model = mjx.put_model(model)
        self.hold = control_hold(self.dt)
        self.n_steps = int(seconds / self.dt)
        self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.reach = morph.max_reach
        self.robot = jnp.asarray(model.geom_contype == 2)
        tgeoms, self.twheel = terminal_geoms(model, morph)
        self.tgeoms = jnp.asarray(tgeoms, dtype=int)
        self.d0 = mjx.forward(self.model, mjx.make_data(self.model))
        self._run = jax.jit(jax.vmap(self._episode))

    def _episode(self, omega, amp1, phase1, offset, amp2, phase2, sigma):
        mx, torso, dt = self.model, self.torso, self.dt
        x0 = self.d0.xpos[torso, 0]
        y0 = self.d0.xpos[torso, 1]
        yaw0 = jnp.arctan2(self.d0.xmat[torso, 1, 0], self.d0.xmat[torso, 0, 0])

        def body(carry, k):
            d, theta, held, dx, dy, energy, duration, done, unstable, touch, zsum, upsum, ticks, yaw, prev, x_mid, reached = carry
            t = d.time
            ramp = jnp.minimum(1.0, t / RAMP_TIME)
            new = ramp * (offset + amp1 * jnp.sin(theta + phase1)
                          + amp2 * jnp.sin(2.0 * theta + phase2))
            held = jnp.where(k % self.hold == 0, new, held)   # 50 Hz hold
            d = mjx.step(mx, d.replace(ctrl=held))
            load = jnp.clip(d.actuator_force / SERVO_TORQUE, -1.0, 1.0)
            theta = theta + dt * (omega - sigma * load * jnp.cos(theta))
            work = jnp.abs(d.actuator_force * d._impl.actuator_velocity).sum() * dt
            blew_up = ~jnp.isfinite(d.qacc).all() | (jnp.abs(d.qvel).max() > 1e3)
            ended = blew_up | (d.xmat[torso, 2, 2] < UPRIGHT_MIN)
            first = ended & ~done
            dx = jnp.where(done, dx, d.xpos[torso, 0] - x0)
            dy = jnp.where(done, dy, d.xpos[torso, 1] - y0)
            energy = jnp.where(done, energy, energy + work)
            # Same as Episode: accumulate wrapped per-step heading differences.
            now = jnp.arctan2(d.xmat[torso, 1, 0], d.xmat[torso, 0, 0])
            yaw = jnp.where(done, yaw, yaw + wrap_angle(now - prev))
            duration = jnp.where(done, duration, d.time)
            unstable = jnp.where(first, blew_up, unstable)
            # Behaviour is sampled once per control tick, while the episode runs.
            sample = ((k + 1) % self.hold == 0) & ~(done | ended)
            touch = touch + sample * self._touching(d)
            zsum = zsum + sample * d.xpos[torso, 2]
            upsum = upsum + sample * d.xmat[torso, 2, 2]
            ticks = ticks + sample
            # Half-way mark, as in Episode: only if the episode is still running.
            at_mid = (k + 1 == self.n_steps // 2) & ~(done | ended)
            x_mid = jnp.where(at_mid, d.xpos[torso, 0] - x0, x_mid)
            reached = reached | at_mid
            return (d, theta, held, dx, dy, energy, duration, done | ended, unstable,
                    touch, zsum, upsum, ticks, yaw, now, x_mid, reached), None

        carry = (self.d0, jnp.zeros_like(amp1), jnp.zeros_like(amp1), 0.0, 0.0, 0.0, 0.0,
                 False, False, jnp.zeros(len(self.tgeoms)), 0.0, 0.0, 0, 0.0, yaw0, 0.0, False)
        (d, _, _, dx, dy, energy, duration, done, unstable, touch, zsum, upsum,
         ticks, yaw, _, x_mid, reached), _ = (
            jax.lax.scan(body, carry, jnp.arange(self.n_steps)))
        # Same convention as Episode: a blow up reports no displacement.
        keep = jnp.where(unstable, 0.0, 1.0)
        # As in Episode, a blow up is reported as unstable, not as a fall.
        # progress = 2 * (x_end - x_mid) if the half-way step was reached, else dx.
        progress = jnp.where(reached, 2.0 * (dx - x_mid), dx)
        return (dx * keep, dy * keep, energy, done & ~unstable, unstable, duration,
                touch, zsum, upsum, ticks, yaw * keep, progress * keep)

    def _touching(self, d):
        """Per terminal point: any active contact with a non-robot geom."""
        c = d._impl.contact
        a, b = c.geom[:, 0], c.geom[:, 1]
        active = c.dist < 0
        t = self.tgeoms[:, None]
        hit = ((a[None] == t) & ~self.robot[b][None]) | ((b[None] == t) & ~self.robot[a][None])
        return jnp.any(hit & active[None], axis=1)

    def run_batch(self, ctrls: np.ndarray, command: float = 0.0) -> List[Metrics]:
        ctrls = np.asarray(ctrls, dtype=float).reshape(-1, CTRL_LENGTH)
        n = len(ctrls)
        assert n > 0
        pad = -n % BATCH_ALIGN
        if pad:
            ctrls = np.concatenate([ctrls, np.repeat(ctrls[-1:], pad, axis=0)])
        ps = [Params(c, self.layout, command) for c in ctrls]

        def stack(name):
            return jnp.asarray(np.stack([getattr(p, name) for p in ps]))

        out = self._run(
            jnp.asarray([p.omega for p in ps]),
            *(stack(n) for n in ("amp1", "phase1", "offset", "amp2", "phase2",
                                 "sigma")),
        )
        dx, dy, energy, fell, unstable, duration, touch, zsum, upsum, ticks, yaw, progress = (
            np.asarray(v) for v in out)
        beh = [behaviour(touch[i], self.twheel, float(zsum[i]), int(ticks[i]), self.reach)
               for i in range(n)]
        return [
            Metrics(dx=float(dx[i]), dy=float(dy[i]), energy=float(energy[i]),
                    fell=bool(fell[i]), unstable=bool(unstable[i]),
                    duration=float(duration[i]), mass=self.mass,
                    duty=beh[i][0], height=beh[i][1], wheel_share=beh[i][2],
                    yaw=float(yaw[i]), progress=float(progress[i]),
                    upright=float(upsum[i] / ticks[i]) if ticks[i] else 1.0)
            for i in range(n)
        ]

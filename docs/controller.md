# Controller and steering

The controller genome and the oscillator that drives each joint. The steering command is described in [Fitness and tasks](fitness-tasks.md).

## Controller genome (`darwinbot/controller.py`)

Length: `CTRL_LENGTH = 1 + MAX_NODES * 6 = 121`, indexed by the same node slot ids.

| Index | Gene | Range | Meaning |
|-------|------|-------|---------|
| 0 | `frequency` | 0.3 - 1.5 Hz | global natural frequency, `omega = 2*pi*frequency` |

Per node slot `s`, with base `b = 1 + 6 * s`:

| Offset | Gene | Range | Meaning |
|--------|------|-------|---------|
| +0, +1 | `amp1`, `phase1` | 0 - 0.8 rad, 0 - 2*pi | first harmonic |
| +2 | `offset` | -0.6 - 0.6 rad | joint centre (for a wheel: signed speed, -4.71 - 4.71 rad/s) |
| +3, +4 | `amp2`, `phase2` | 0 - 0.8 rad, 0 - 2*pi | second harmonic |
| +5 | `sigma` | -3 - 3 rad/s | load feedback gain (Tegotae) |

Every joint has its own phase oscillator, driven by its servo load:

```
dtheta_i/dt = omega - sigma_i * N_i * cos(theta_i)
target_i    = ramp(t) * (offset_i + amp1_i * sin(theta_i + phase1_i)
                                  + amp2_i * sin(2 * theta_i + phase2_i))
```

`N_i` is the servo load divided by the maximum torque, between -1 and 1: the
actuator force in simulation and the STS3215 load register on the real robot.
Legs that carry weight slow down or speed up their own phase, so coordinated
gaits can appear without programmed coupling between legs (Tegotae, Owaki et
al.). With `sigma = 0` it is an open-loop CPG. `ramp` fades the gait in over
`RAMP_TIME = 0.5 s`. Genes of absent modules are ignored, and a wheel only uses
its `offset` gene. The total amplitude is clipped so that
`(amp1 + 2 * amp2) * omega <= 4.71 rad/s`, the servo speed limit. The
simulation updates the servo targets at 50 Hz and holds them in between, like
the ESP32.

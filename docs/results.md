# Results

The showcase run uses the default `steer` task on the default `terrain` set
(flat, rough, steps and rubble), with two randomised physics variants per
evaluation (`--preset steer --batches 30`). It ran on Kaggle CPU sessions
(4 cores each), about 40 hours in four chained sessions, and made 124,874
evaluations, each one three episodes (straight, left, right) on every terrain.
It filled 79 of the ~500 cells, every one of them buildable:

| Gait | Types | Best fitness |
|------|-------|--------------|
| `roll` | 26 | 0.43 |
| `trot` | 28 | 0.29 |
| `walk` | 12 | 0.17 |
| `crawl` | 6 | 0.08 |
| `belly` | 7 | 0.02 |

47 types score above 0.05, which means they move and also turn on command.
Passive modules are common: 32 elites use a TPU spring, 13 a caster, 12 a
tail, 3 a spike and 1 a skid, and 5 have a second torso segment. Link lengths
span the whole 40 to 200 mm the genome allows.

The best roller (`16servos-low-roll-389`), the best trotter (`14servos-mid-trot`)
and a five-servo trotter (`5servos-low-trot-364`) all pass `validate.py`, with
mean losses of 0%, 6% and 13% under randomised physics (the limit is 30%).

At batch 25 a robot that spun on the spot on one wheel scored 1.99, because
the turn credit had no ceiling. The turn credit is now counted up to 90
degrees (`TURN_CAP` in `fitness.py`), and `experiments/rescore_archive.py`
rescored every elite under the new rule, keeping cells and lineage; the dotted
line in the progress plot marks that point.

## Steering versus forward only

Turning on command costs straight-line speed. A control run with the same
terrains, variants and number of batches, trained on the `forward` task
(`--preset forward`), was scored next to the showcase run under the forward
rule:

| | `steer` (showcase) | `forward` (control) |
|---|---|---|
| Evaluations | 124,874 | 127,938 |
| Cells | 79 | 84 |
| Best roller | 0.64 | 0.81 |
| Best trotter | 0.54 | 0.55 |
| Best walker | 0.31 | 0.46 |
| Best belly crawler | 0.06 | 0.26 |

Trotters barely pay for steering, while rollers, walkers and belly crawlers
lose a fifth to three quarters of their forward speed. The control robots
cannot turn, and a robot you build needs to. Use `--preset forward` when
turning on command does not matter; it is also about three times cheaper per
evaluation.

## Earlier forward-only runs

Before steering and the free-form shapes, the project trained only on the
`forward` task on flat, rough and steps. The first four columns are 6-batch
runs from before the physics review that fixed hidden simulator resets and
passive joint friction; the last is a 30-batch laptop run after it. They used
an older scoring rule and a smaller search space, so they compare with the
runs above only roughly:

| | Leg pairs | Tree bodies | Tree + cost of transport | Free form | Free form, 30 batches on a laptop |
|---|---|---|---|---|---|
| Robot types | 20 | 50 | 51 | 46 | 98 |
| Types that move forward | 14 | 25 | 21 | 28 | 64 |
| Best rolling robot | 1.27 | 0.95 | 0.91 | 1.01 | 1.03 |
| Best legged robot | 0.40 | 0.39 | 0.25 | 0.15 | 0.25 |

![Evolved body plans](img/robots.png)

![MAP-Elites archive](img/archive_map.png)

![Training progress](img/progress.png)

![Printable parts of the best robot](img/parts.png)

## What counts as success (honestly)

| Level | What it means | Promised? |
|-------|---------------|-----------|
| **Minimum** | Diverse robots that walk or roll in simulation, built only from real parts | Yes |
| **Good** | An elite that passes `validate.py` (keeps most of its distance on randomised physics) and exports a coherent kit | Yes, with a full training run |
| **Moonshot** | A printed and assembled robot that walks with its evolved gait | No, it needs the hardware |

The value is in the pipeline being correct end to end, from genome to parts
list, not in any single robot.

## Known limitations

- Wheels still lead on speed: the best roller scores about 1.5 times the best
  trotter, and the top of the archive is mostly `roll`.
- The firmware is half done: `firmware/core/` runs the controller math in portable C++ and a test proves it matches the Python controller tick by tick (`tests/test_firmware_parity.py`). The ESP32 layer (servo bus, load reading, calibration) waits for real hardware.
- Randomised training (`--variants`) is CPU only.
- The only feedback is the servo load. The controller turns when commanded but
  never sees the world, so tasks that need reacting (finding a doorway,
  avoiding a wall) are out of scope.
- A body whose parts overlap at spawn, whose parts exceed the 220 mm print bed
  or whose links bend beyond 25 MPa under the servo stall torque is not
  buildable and scores `UNSTABLE_SCORE` without being simulated (common among
  random trees).
- Spring stiffness comes from a
  TPU modulus that is not verified, and the printability stress check is a
  cantilever estimate, not a finite-element analysis.

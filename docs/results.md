# Results

## Showcase

The current showcase comes from the long run with 12 V servos, the
assembly checks and plastic friction. Five CPU islands on Kaggle, one night
island on a desktop and one GPU island trained on the `steer` task and were
merged (129 cells, best fitness 0.56). The best robots of each region then
got a longer controller search (`experiments/polish.py`), lost the parts they
did not need (`experiments/prune.py`), passed `experiments/validate.py` (mean
loss between -18% and 5% under varied physics, limit 30%) and
`tools/check_assembly.py` (no clashes between printed parts or servo bodies):

| Robot | Fitness | Before polishing |
|---|---|---|
| 13servos-mid-trot | 0.71 | 0.46 (a roller) |
| 16servos-mid-roll | 0.70 | 0.56 |
| 13servos-mid-belly | 0.66 | 0.33 |
| 6servos-low-trot | 0.53 | 0.36 (7 servos before pruning) |
| 6servos-low-belly | 0.54 | 0.40 (9 servos before pruning) |
| 11servos-low-belly | 0.54 | 0.39 |

Polishing changed the gait of two robots: the 13-servo body trotted on four
legs once its controller had more search. Walking and crawling robots stay
weak (0.10 to 0.12): their links scrape the floor and their joints stop
early.

![Evolved body plans](img/robots.png)

![Printable parts of the best robot](img/parts.png)

The archive of these robots is attached to the
[showcase release](https://github.com/mpena2099/darwinbot/releases/tag/showcase-1):

```bash
curl -LO https://github.com/mpena2099/darwinbot/releases/download/showcase-1/showcase.pkl
python experiments/replay.py --archive showcase.pkl --list
python experiments/replay.py --archive showcase.pkl --type 13servos-mid-trot --speed 0.5
python experiments/export.py --archive showcase.pkl --type 13servos-mid-trot
```

## Previous steer run

This archive uses the default `steer` task on the default `terrain`
set (flat, rough, steps and rubble), with two randomised physics variants per
evaluation (`--preset steer --batches 30`). Three runs of 30 batches with
different seeds trained as islands on Kaggle and were merged
(`experiments/merge_archives.py`), each cell keeping the better elite:

| | CPU island 1 | CPU island 2 | GPU island |
|---|---|---|---|
| Hardware | 4-core CPU sessions | 4-core CPU sessions | one T4, MuJoCo Warp (`--device warp`, branch `warp`) |
| Training time | about 40 h | 37.1 h | 15.4 h |
| Evaluations | 124,874 | 142,318 | 135,824 |
| Cells | 73 | 83 | 76 |
| Best fitness | 0.41 | 0.60 | 0.50 |
| QD-score | 368 | 418 | 383 |

The first island trained before the turn credit was capped by the distance
travelled and was rescored under the current rule
(`experiments/rescore_archive.py`). The GPU trains 2.3 times faster than the
second CPU island for the same quality; the gaps in best fitness are within
what a different seed gives, and each island leads somewhere: the second CPU
island has the best roller, the first the best trotter and walker.

Before the merge, 16 elites (6, 2 and 8) were removed because a part sat
deeper than 29 mm inside the part it is mounted on, such as a wheel bolted
inside the torso edge. MuJoCo never collides a body with its parent, so the
simulator let them through; `is_buildable` now checks it.

The merged archive has 403,016 evaluations, each one three episodes (straight,
left, right) on every terrain, and fills 116 of the ~500 cells:

| Gait | Types | Best fitness |
|------|-------|--------------|
| `roll` | 31 | 0.60 |
| `trot` | 35 | 0.29 |
| `walk` | 23 | 0.17 |
| `belly` | 17 | 0.07 |
| `crawl` | 10 | 0.07 |

57 types score above 0.05, which means they move and also turn on command.
Passive modules are common: 23 elites use a TPU spring, 12 a tail, 7 a
caster, 6 a skid and 6 a spike, and 9 have a second torso segment. Link
lengths span the whole 40 to 200 mm the genome allows.

The four robots below (the best roller `16servos-low-roll-389`, a two-wheeler
`8servos-low-roll`, the best trotter `14servos-mid-trot` and the best walker
`12servos-low-walk-344`) all pass `validate.py`, with mean losses of -1%, 4%,
6% and 3% under randomised physics (the limit is 30%). The progress plot
follows the second CPU island.

Longer training would not change much: in that island the best roller went
from 0.55 at batch 3 to 0.60 at batch 30, and the best walker has stayed at
0.15 since batch 15.

## Steering versus forward only

A control run with the same terrains and variants, 30 batches on the `forward`
task (`--preset forward`, one CPU run), was compared with the previous steer archive
scored under the forward rule:

| | `steer` (previous run) | `forward` (control) |
|---|---|---|
| Evaluations | 403,016 (three islands) | 127,938 |
| Cells | 116 | 78 |
| Best roller | 0.93 | 0.81 |
| Best trotter | 0.54 | 0.53 |
| Best walker | 0.31 | 0.34 |
| Best belly crawler | 0.19 | 0.26 |

The steer run had three times the evaluations. Its rollers went further, its
trotters matched the control and its walkers came close, while also turning
on command. The control robots cannot turn, and a
robot you build needs to. Use `--preset forward` when turning on command does
not matter; it is also about three times cheaper per evaluation.

## How the environment shapes the body

Two short runs (9 batches, about 30,000 evaluations each, same rule and
variants) differ only in their terrains: one trained on flat ground alone,
the other on the four terrains. Mean `steer` score of each run's top 10:

| Trained on | flat | rough | steps | rubble | Servos | Wheels |
|---|---|---|---|---|---|---|
| flat only | 0.50 | 0.26 | 0.02 | 0.31 | 10.3 | 2.9 |
| four terrains | 0.57 | 0.48 | 0.25 | 0.46 | 13.8 | 3.1 |

The terrain-trained robots are better even on flat ground. Robots that never
saw the step stop at it; those trained on it carry more servos and climb it.

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

![MAP-Elites archive](img/archive_map.png)

![Training progress](img/progress.png)

## What counts as success (honestly)

| Level | What it means | Promised? |
|-------|---------------|-----------|
| **Minimum** | Diverse robots that walk or roll in simulation, built only from real parts | Yes |
| **Good** | An elite that passes `validate.py` (keeps most of its distance on randomised physics) and exports a coherent kit | Yes, with a full training run |
| **Moonshot** | A printed and assembled robot that walks with its evolved gait | No, it needs the hardware |

The value is in the pipeline being correct end to end, from genome to parts
list, not in any single robot.

## Known limitations

- Wheels still lead on speed: the best roller scores about 3 times the best
  trotter, and the top of the archive is mostly `roll`.
- The firmware is half done: `firmware/core/` runs the controller math in portable C++ and a test proves it matches the Python controller tick by tick (`tests/test_firmware_parity.py`). The ESP32 layer (servo bus, load reading, calibration) waits for real hardware.
- Randomised training (`--variants`) is CPU only.
- The only feedback is the servo load. The controller turns when commanded but
  never sees the world, so tasks that need reacting (finding a doorway,
  avoiding a wall) are out of scope.
- A body whose parts overlap at spawn or sink more than 29 mm into the part
  they are mounted on, whose parts exceed the 220 mm print bed
  or whose links bend beyond 25 MPa under the servo stall torque is not
  buildable and scores `UNSTABLE_SCORE` without being simulated (common among
  random trees).
- Spring stiffness comes from a
  TPU modulus that is not verified, and the printability stress check is a
  cantilever estimate, not a finite-element analysis.

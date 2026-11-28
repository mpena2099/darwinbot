# Results

This page says what the project has achieved and how it was measured. The
numbers come from the run called long8, the showcase-2 release. Showcase-1
robots are not part of these tables, but they still play (see "Playing the
robots").

## How it is measured

- Every episode is 8 s of simulated time. A robot is judged on how far it
  progresses in the second half of the episode, and a fall before half-way
  scores no progress.
- The task is `steer`. Each evaluation runs four episodes: straight, left,
  right and half speed. The score is 0.4 for going straight, 0.4 for turning
  (the mean of left and right) and 0.2 for half speed.
- The archive is a CVT archive of 800 cells over five axes: duty, torso
  height, servo count, number of legs and cost of transport.
- A robot passes `experiments/validate.py` when its mean loss under varied
  physics is below 30%, its 5th percentile loss is below 60% and at most 10%
  of the variants fall. Every terrain must pass, not only the average.
  `--unseen` adds terrains the archive was not trained on.
- A robot passes `tools/check_assembly.py` when no printed part or servo
  body clashes with another and every horn hole lines up.
- Bodies that break a printing rule or overlap at spawn are not simulated.

## Showcase

long8 trained as islands on Kaggle and on two local machines for about three days, from 2026-11-13 to 2026-11-16.
The hardware was five Kaggle CPU sessions, one Kaggle GPU session with two T4 cards and two local machines. The task is `steer`, the episodes are 8 s and
the body model is 23.

The archive comes in four island settings. Their robots were scored on
different ground or with different parts, so the archives cannot be merged
with each other:

| Setting | Terrains | Wheels |
|---|---|---|
| Terrain preset | flat, rough, steps and rubble | allowed |
| Flat | flat only | allowed |
| Flat and steps | flat and steps | allowed |
| No wheels | terrain preset | not allowed |

Islands with the same setting merge with `experiments/merge_archives.py`,
which keeps the best elite of each cell. Each setting keeps its own archive
of 800 cells:

| Setting | Cells filled | Body types | With legs |
|---|---|---|---|
| Terrain preset | 164 | 72 | 131 |
| Flat | 193 | 84 | 162 |
| Flat and steps | 160 | 73 | 135 |
| No wheels | 85 | 45 | 74 |

![Evolved body plans](img/robots.png)

![Four showcase robots moving](img/gait.gif)

### Best robots

The five showcase robots, picked by eye from the best of each setting. Each
went through the longer controller search (`experiments/polish.py`) and the
removal of parts it does not need (`experiments/prune.py`); the table names
the version published, the one that passes validation where both exist.

| Setting | Robot | Servos | Legs | Wheels | Fitness | Version |
|---|---|---|---|---|---|---|
| Terrain preset | `5servos-low-roll` | 5 | 2 | 2 | 0.57 | polished |
| Flat | `3servos-low-roll` | 3 | 0 | 2 | 1.15 | pruned |
| Flat | `10servos-mid-belly` | 10 | 5 | 0 | 1.07 | polished |
| Flat and steps | `10servos-mid-belly` | 10 | 5 | 0 | 0.32 | pruned |
| No wheels | `6servos-low-belly-498` | 6 | 2 | 0 | 0.26 | as trained |

![Printable parts of the best robot](img/parts.png)

### Validation

The robots above, run through `experiments/validate.py`:

| Robot | Validation | Weak spot |
|---|---|---|
| `5servos-low-roll` (preset) | PASS | barely moves on rubble even with nominal physics; its pruned version fails rubble turning right |
| `3servos-low-roll` (flat) | PASS | |
| `10servos-mid-belly` (flat) | PASS | does not turn right |
| `10servos-mid-belly` (flat and steps) | PASS | |
| `6servos-low-belly-498` (no wheels) | PASS | does not turn right; the most robust, at most 26% p5 loss |

The no-wheels robot first picked, `13servos-low-belly-30`, failed on rough
ground, steps and rubble when turning, and was replaced.

### Turning

Turning is weaker than going straight. Two of the five showcase robots do not turn right at all and score only by going straight, left and at half speed.

The straight episode and the turn episodes share one controller, and a turn
is a command added to the same oscillators. A robot that walks well has
nothing in its gait that guarantees a good turn.

## Playing the robots

The archive of the showcase-2 robots is attached to the
[showcase-2 release](https://github.com/mpena2099/darwinbot/releases/tag/showcase-2),
with a snapshot of their gaits that any later version of the code can play.

```bash
curl -LO https://github.com/mpena2099/darwinbot/releases/download/showcase-2/showcase.snap.pkl
python experiments/replay.py --snapshot showcase.snap.pkl --list
python experiments/replay.py --snapshot showcase.snap.pkl --type <type> --speed 0.5
```

The showcase-1 robots (body model 20) still play from their own snapshot,
which needs no migration:

```bash
curl -LO https://github.com/mpena2099/darwinbot/releases/download/showcase-1/showcase.snap.pkl
python experiments/replay.py --snapshot showcase.snap.pkl --list
```

To export one of them, or to train on its archive, bring the archive up to
the current body model first. `experiments/migrate_archive.py` simulates
every robot again under the current rules, searches each controller again
and drops bodies that are no longer buildable
([resuming after a code change](long-runs.md#resuming-after-a-code-change)).

## Training it yourself

Run the `steer` preset (the `steer` task on the full terrain set; plain
`train.py` trains on flat ground only) as islands, then take the chosen
robots through
[the showcase steps](long-runs.md#from-an-archive-to-a-showcase):

```bash
python tools/kaggle/islands.py --user <you> --code-dataset <you>/darwinbot-code \
    --islands 5 --rounds 4 --batches-per-round 8 --train-args "--preset steer"
# or on one machine (much slower):
python experiments/train.py --preset steer --batches 200 --workers 8
```

`tools/kaggle/islands.py` takes the terrains and body options of each
island, which is how the four settings above were made. A different seed
gives different robots.

## What the archive looks like

![MAP-Elites archive](img/archive_map.png)

![Training progress](img/progress.png)

The map shows the cells of one setting, coloured by fitness. The progress
plot follows the best fitness and the number of filled cells over the
batches of one island.

## What counts as success

| Level | What it means | Reached? |
|-------|---------------|----------|
| **Minimum** | Diverse robots that walk or roll in simulation, built only from real parts | Yes |
| **Good** | An elite that passes `validate.py` (keeps most of its distance on randomised physics) and exports a coherent kit | Yes |
| **Moonshot** | A printed and assembled robot that walks with its evolved gait | No, it needs the hardware |

The value is in the pipeline being correct end to end, from genome to parts
list, not in any single robot.

## Known limitations

- Wheels win where they are allowed. Rolling robots fill 91 of 164 cells in the terrain preset, 78 of 193 on flat and 65 of 160 on flat and steps, and the best robot of each of those settings rolls. The no-wheels setting exists to keep legged gaits in the running. A robot is named `roll` when its wheels hold most of the ground contact, even if its torso drags too; release 3 measures the torso contact and names those robots `belly`.
- The firmware is written but not yet run: `firmware/core/` holds the controller math in portable C++ with a test that matches the Python controller tick by tick (`tests/test_firmware_parity.py`), and `firmware/esp32/` puts it on the robot (servo bus, load reading, web page, micro-ROS). No robot exists yet, so neither has touched hardware.
- Nothing has been built yet. Every number on this page comes from
  simulation, and the servo measures come from a STEP model, not from a
  caliper.
- The only feedback is the servo load. The controller turns when commanded
  but never sees the world, so tasks that need reacting (finding a doorway,
  avoiding a wall) are out of scope.
- A body whose parts overlap at spawn or sink more than 29 mm into the part
  they are mounted on, whose parts exceed the print bed or whose links bend
  beyond 25 MPa under the servo stall torque is not buildable and scores
  `UNSTABLE_SCORE` without being simulated (common among random trees).
- Spring stiffness comes from a TPU modulus that is not verified, and the
  printability stress check is a cantilever estimate, not a finite-element
  analysis.
- Archives of the four settings cannot be merged, so there is no single
  best robot, only the best of each setting.

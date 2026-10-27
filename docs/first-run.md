# First run

Train a small archive, list the robot types it found and watch one move.

```bash
python experiments/train.py
python experiments/replay.py --list
python experiments/replay.py --type 8servos-mid-trot
```

## All commands

```bash
# Evolve on the flat floor with the default settings
python experiments/train.py

# Flat, rough, steps and rubble, with more batches and workers
python experiments/train.py --terrains terrain --batches 60 --workers 8

# Learn to turn in place instead of walking forward
python experiments/train.py --task turn

# Only some environments (see Environments and presets)
python experiments/train.py --terrains rubble,soft

# Climb a 3 cm step, or carry 0.3 kg on the torso
python experiments/train.py --terrains step
python experiments/train.py --payload 0.3

# Resume a previous run (task and payload come from the archive)
python experiments/train.py --resume checkpoints/archive.pkl --batches 20

# Draw parents by novelty (default) or uniformly at random
python experiments/train.py --parent-selection uniform

# Replay the best elite in the MuJoCo viewer (flat terrain)
python experiments/replay.py

# List the robot types found so far, then open one of them
python experiments/replay.py --list
python experiments/replay.py --type 8servos-mid-trot --terrain rough

# Record a video instead of opening a window (needs the video extra)
python experiments/replay.py --record out.mp4 --seconds 8

# Train on randomised models (default 2 per terrain on cpu, 0 on gpu)
python experiments/train.py --variants 4

# Check the elite on 50 randomised models per terrain (PASS if mean loss < 30%)
python experiments/validate.py --n 50

# Export a robot to build: parts, bill of materials, assembly, control file
python experiments/export.py --type 8servos-mid-trot

# Plot the behaviour scatter and the progress curves
python experiments/plot_archive.py
python experiments/plot_archive.py --show

# Family tree: ancestors of one robot, or ancestor variety against fitness
python experiments/lineage.py --type 8servos-mid-trot
python experiments/lineage.py
```

Training writes `checkpoints/archive.pkl` after every batch. `replay.py` reads
that file by default (`--archive` changes it). `plot_archive.py` saves
`archive_heatmap.png` (the behaviour scatter plot: one dot per elite at its duty
and height, coloured by fitness, sized by servo count) and `archive_history.png`
to `checkpoints/` (`--out` changes the directory).

`--variants N` scores every candidate gait on N randomised copies of each
terrain (friction, mass, servo torque and stiffness, joint play, 0 to 2
control periods of command delay) and takes the mean, so gaits that depend on
exact simulator details lose. It costs about N times more per evaluation, is
CPU only, and is stored in the archive. `validate.py` runs the elite on
`--n` fresh variants (`--seed`, `--workers`) and prints nominal, mean and
worst distance, fall rate and relative loss per terrain.

## Viewing the different robot types

The archive keeps one robot per **cell** of the behaviour map: the best one found
with that duty, height, number of legs and cost of transport (four descriptors). A new robot only replaces the one in
its own cell, and only if it scores higher, so slow crawlers survive next to
fast rollers.

A type is a short name derived from the cell, `<servos>servos-<height>-<gait>`
(for example `8servos-mid-trot`):

| Part | Values | Meaning |
|------|--------|---------|
| `servos` | 1 to 16 | number of servos, the cost of the robot (shown in the name, not a map axis) |
| `height` | `low`, `mid`, `high` | mean torso height over body reach: below 0.4, below 0.8, above |
| `gait` | `crawl`, `walk`, `trot`, `belly`, `roll` | duty, the share of time the tips touch the ground: above 0.8, 0.5 to 0.8, 0.15 to 0.5, below 0.15 (sliding on the body); `roll` when wheels carry more than half of the contact time |

Several cells can share a name; the later ones get the cell index appended
(`12servos-low-walk-133`).

Step 1, see which types the run has found, best first:

```
$ python experiments/replay.py --list
type                       fitness  duty  height  servos    cot
8servos-mid-trot              0.11  0.34    0.45       8    6.2
12servos-low-trot            -0.03  0.41    0.28      12   21.4
12servos-low-walk-133        -0.03  0.52    0.33      12   18.7
...
```

Step 2, open one in the 3D viewer (or add `--record name.mp4` for a video):

```bash
python experiments/replay.py --type 8servos-mid-trot
```

Without `--type`, `replay.py` shows the best robot overall.

## Viewer controls

`replay.py` without `--record` opens the interactive MuJoCo viewer: left drag
rotates the camera, right drag pans, scroll zooms, double click selects a
body and Ctrl + double click makes the camera follow it, space pauses, Tab
hides the side panels. `--speed 0.25` plays in slow motion (the window has
no speed control of its own). The simulation does not reset, so the robot eventually
leaves the terrain.

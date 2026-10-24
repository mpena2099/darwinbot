# darwinbot

Quality-diversity evolution of robot bodies and locomotion controllers.

Every robot is a box torso with up to three bilaterally mirrored leg pairs. Each
pair has a hip yaw joint, a hip pitch joint and a knee, plus an optional wheel at
the shin tip. Bodies are evolved with MAP-Elites while a central pattern
generator (CPG) is tuned with CMA-ES for each candidate body.

## In one paragraph

The program invents robots and teaches each one to move. A robot is a box with
legs; evolution decides how many legs, how long, where they attach and whether
they end in a wheel. For every new body, an optimiser searches for the rhythm
of joint movements that carries it farthest forward in a few simulated seconds.
The run does not keep a single winner: it keeps the best robot of each *kind*
(two-legged with wheels, six-legged without wheels, long legs, short legs...),
so at the end you can browse a catalogue of different solutions.

## How it works

Two nested loops:

1. **Outer loop, MAP-Elites over bodies.** The archive is a pyribs GridArchive
   indexed by the number of leg pairs, the number of wheeled pairs and a
   leg-length bin. Each cell keeps the best body found with those features, so
   a run ends with a map of specialists (best walker, best wheeler, best
   long-legged hexapod) instead of a single design.
2. **Inner loop, CMA-ES over a CPG.** For a fixed body, CMA-ES searches the
   controller genes that this body actually uses. A promising body is therefore
   judged by the best gait it can learn, not by a lucky random controller.

Children are mutated copies of random elites. The child starts its inner loop
from the parent controller, so learned gaits are inherited along with the body
(Lamarckian inheritance). Only the genes that affect the child are optimised;
the rest stay in the genome and become active again if a pair or wheel returns.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
```

Dependencies: `mujoco >= 3.6`, `numpy >= 1.23`, `cma >= 3.3`,
`matplotlib >= 3.6`. `imageio` is optional and only needed for video recording:

```bash
pip install -e ".[video]"      # or: pip install imageio[ffmpeg]
```

## Run

```bash
# Evolve on the flat floor with the default settings
python experiments/train.py

# Flat, rough and steps, with more batches and workers
python experiments/train.py --terrains terrain --batches 60 --workers 8

# Resume a previous run
python experiments/train.py --resume checkpoints/archive.pkl --batches 20

# Replay the best elite in the MuJoCo viewer (flat terrain)
python experiments/replay.py

# List the robot types found so far, then open one of them
python experiments/replay.py --list
python experiments/replay.py --type 6legs-2wheels-14servos --terrain rough

# Record a video instead of opening a window (needs the video extra)
python experiments/replay.py --record out.mp4 --seconds 8

# Plot the archive heatmap and the progress curves
python experiments/plot_archive.py
python experiments/plot_archive.py --show
```

Training writes `checkpoints/archive.pkl` after every batch. `replay.py` reads
that file by default (`--archive` changes it). `plot_archive.py` saves
`archive_heatmap.png` and `archive_history.png` to `checkpoints/` (`--out`
changes the directory).

### GPU (optional)

The inner CMA-ES loop can run on an NVIDIA GPU through MJX. The CPU remains the
default and works without JAX installed.

```bash
pip install -e ".[gpu]"
python experiments/train.py --device gpu --popsize 128
```

On the GPU, bodies are optimised one at a time and `--workers` is ignored (one
device, and forking with CUDA does not work). `--popsize` defaults to 64 on the
GPU. A strong NVIDIA GPU is needed to beat the CPU: the GeForce 940MX on this
machine is about 12x slower, see `docs/plano-gpu.md`.

### Viewing the different robot types

The archive keeps one robot per **type**: the best one found with that number
of legs, number of wheels and number of servos (the cost of the robot). A new robot only replaces the one
of its own type, and only if it scores higher, so slow walkers survive next to
fast wheeled robots.

A type is named `<legs>legs-<wheels>wheels-<servos>servos`:

| Part | Values | Meaning |
|------|--------|---------|
| `legs` | 2, 4, 6 | number of legs (always in mirrored pairs) |
| `wheels` | 0 up to `legs` | how many legs end in a wheel |
| `servos` | 4, 6, ..., 24 | total servos: hip pitch and knee per leg, plus one per yaw joint and one per wheel |

Step 1, see which types the run has found, best first:

```
$ python experiments/replay.py --list
type                    fitness  leg length  servos
2legs-2wheels-6servos      0.85  0.16 m     6
4legs-4wheels-12servos     0.60  0.20 m     12
6legs-0wheels-18servos    -0.06  0.17 m     18
...
```

Step 2, open one in the 3D viewer (or add `--record name.mp4` for a video):

```bash
python experiments/replay.py --type 6legs-2wheels-14servos
```

Without `--type`, `replay.py` shows the best robot overall.

### Viewer controls

`replay.py` without `--record` opens the interactive MuJoCo viewer: left drag
rotates the camera, right drag pans, scroll zooms, double click selects a
body and Ctrl + double click makes the camera follow it, space pauses, Tab
hides the side panels. `--speed 0.25` plays in slow motion (the window has
no speed control of its own). The simulation does not reset, so the robot eventually
leaves the terrain.

### Long runs

A run on the `terrain` preset with 3 workers takes roughly 15 to 20 min for
the initial batch and 3 to 4 min for each later batch (5 s episodes,
0.004 s physics step, 120 CMA-ES evaluations per child; see `--seconds`,
`--budget-init` and `--budget-child`). The archive is saved after
every batch; stopping mid-batch loses only that batch. Keep the machine awake
and the process alive when the terminal closes:

```bash
tmux new -s darwin
systemd-inhibit --what=sleep python experiments/train.py --terrains terrain --batches 40 --workers 3
# detach with Ctrl+b d, reattach with: tmux attach -t darwin
```

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## Genome

All genes are normalised to `[0, 1]` and mapped linearly onto their physical
range on decode, so one mutation sigma works for every gene.

### Body genome (`darwinbot/morphology.py`)

Length: `MORPH_LENGTH = 2 + N_PAIRS * 8 = 26` with `N_PAIRS = 3`.

| Index | Gene | Range | Meaning |
|-------|------|-------|---------|
| 0 | `body_length` | 0.15 - 0.40 m | torso length |
| 1 | `body_width` | 0.10 - 0.25 m | torso width |

Per pair `p`, with base `b = 2 + 8 * p`:

| Offset | Gene | Range | Meaning |
|--------|------|-------|---------|
| +0 | `enabled` | 0 - 1 | pair exists when > 0.5 |
| +1 | `attach_x` | -1.0 - 1.0 | position along the torso side, rear to front |
| +2 | `splay` | -0.6 - 0.6 rad | leg yaw at rest |
| +3 | `thigh_len` | 0.06, 0.08, 0.10 or 0.12 m | printed link; the gene picks the catalogue index |
| +4 | `shin_len` | 0.06, 0.08, 0.10 or 0.12 m | printed link; same catalogue |
| +5 | `has_yaw` | 0 - 1 | hip has a yaw servo when > 0.5 |
| +6 | `has_wheel` | 0 - 1 | wheel at the shin tip when > 0.5 |
| +7 | `wheel_rad` | 0.03 - 0.06 m | wheel radius |

If no pair is enabled, the pair with the highest `enabled` gene is switched on
so every body has at least one leg pair.

### Controller genome (`darwinbot/controller.py`)

Length: `CTRL_LENGTH = 1 + N_PAIRS * 11 = 34`.

| Index | Gene | Range | Meaning |
|-------|------|-------|---------|
| 0 | `frequency` | 0.5 - 3.0 Hz | global gait frequency |

Per pair `p`, with base `b = 1 + 11 * p`:

| Offset | Gene | Range | Meaning |
|--------|------|-------|---------|
| +0, +1, +2 | `hip_yaw` amp, offset, phase | 0 - 0.8 rad, -0.6 - 0.6 rad, 0 - 2*pi | yaw joint signal (ignored without `has_yaw`) |
| +3, +4, +5 | `hip_pitch` amp, offset, phase | same | pitch joint signal |
| +6, +7, +8 | `knee` amp, offset, phase | same | knee signal |
| +9 | `side_shift` | 0 - 2*pi | right leg phase = left phase + `side_shift` |
| +10 | `wheel_speed` | -5.45 - 5.45 rad/s | signed wheel velocity |

The signal is `ramp(t) * (offset + amp * sin(omega * t + phase))`, where `ramp`
fades the gait in over `RAMP_TIME = 0.5 s`. Genes of absent pairs, absent yaw
joints or absent wheels are ignored. Each joint amplitude is clipped so that
`amp * omega <= 5.45 rad/s`, the servo speed limit. The simulation updates the
servo targets at 50 Hz and holds them in between, like the ESP32.

## Real parts

Every simulated robot is built from parts you can buy or print: Feetech
STS3215 servos at 7.4 V (1.91 N*m stall, 5.45 rad/s, run at 70% of the torque),
60 g per servo, an ESP32 + 2S LiPo + regulator payload of 0.13 kg in the torso,
printed links of 60, 80, 100 or 120 mm, and self-collision between legs. The
sources and the values still to be checked are in `docs/hardware-dados.md`.

## Terrains

| Name | Description |
|------|-------------|
| `flat` | flat floor |
| `rough` | smoothed random bumps, flat near the spawn |
| `steps` | ascending staircase starting at x = 0.6 m |
| `slope` | 8 degree ramp starting at x = 0.5 m |
| `slippery` | floor with friction 0.2 |

Presets: `flat` (flat), `terrain` (flat, rough, steps) and `all` (all five).
`--terrains` also accepts a comma separated list of names.

## Fitness

Per episode, from `darwinbot/fitness.py`:

```
score = dx - 0.5 * |dy| - 0.05 * energy / (m * g) - 0.5 * fell
```

`dx` is forward displacement, `dy` lateral displacement, `energy` the mechanical
work of all actuators, `m` the robot mass and `g = 9.81`. Dividing energy by
weight turns it into metres and keeps heavy and light bodies comparable. A fall
ends the episode but keeps the distance covered, so the fixed penalty only
breaks ties in favour of staying upright. A physics blow-up returns
`UNSTABLE_SCORE = -5.0`.

Multi-terrain runs use a blend of mean and worst score:

```
fitness = 0.7 * mean(scores) + 0.3 * min(scores)
```

## Known limitations

- Wheels dominate. On the current terrains a body that rolls on two large
  wheels scores about 7, while the best six-legged walker scores below 1. The
  3 cm steps are too low to stop a wheel; taller steps would give legs a real
  advantage.
- The controller is open loop: it never senses the ground, so tasks that need
  steering or reacting (finding a doorway, avoiding a wall) are out of scope.
- A body whose legs overlap at spawn is not buildable and scores `UNSTABLE_SCORE`
  without being simulated (about 12% of random genomes).

## Project structure

```
darwinbot/
├── darwinbot/
│   ├── morphology.py   # body genome, decode, mutate, descriptor
│   ├── controller.py   # CPG genome, active genes, signal
│   ├── builder.py      # morphology + terrain -> MJCF
│   ├── terrain.py      # terrain snippets and presets
│   ├── sim.py          # Episode: compile once, replay controllers
│   ├── fitness.py      # episode score and multi-terrain aggregate
│   ├── inner.py        # CMA-ES inner loop
│   └── evolve.py       # MAP-Elites archive and outer loop
├── experiments/
│   ├── train.py        # run evolution
│   ├── replay.py       # viewer or video for an elite
│   └── plot_archive.py # heatmap and progress plots
├── tests/
│   ├── test_morphology.py
│   ├── test_controller.py
│   ├── test_sim.py
│   └── test_evolve.py
├── checkpoints/        # archive.pkl and plots
├── pyproject.toml
└── requirements.txt
```

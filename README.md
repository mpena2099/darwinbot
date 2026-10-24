# darwinbot

[![tests](https://github.com/mpena2099/darwinbot/actions/workflows/tests.yml/badge.svg)](https://github.com/mpena2099/darwinbot/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Quality-diversity evolution of robot bodies and locomotion controllers, built
from real parts so the result can be printed and assembled.

Every robot is a box torso with up to three bilaterally mirrored leg pairs. Each
pair has a hip pitch joint and a knee, an optional hip yaw joint and an optional
wheel at the shin tip, all driven by Feetech STS3215 servos. Bodies are evolved
with MAP-Elites while a central pattern generator (CPG) is tuned with CMA-ES for
each candidate body, and any robot in the archive can be exported as printable
parts, a bill of materials and a control file for an ESP32.

## Resumo em português

O DarwinBot inventa robôs e ensina cada um a andar, numa simulação física
(MuJoCo). A evolução decide o corpo: quantas pernas, de que tamanho, onde se
prendem, se o quadril gira e se a perna termina numa roda. Para cada corpo, um
otimizador (CMA-ES) procura o ritmo de movimento das juntas que leva o robô mais
longe. Em vez de um único vencedor, o programa guarda o melhor robô de cada tipo
(MAP-Elites), formando um catálogo de soluções diferentes. Os robôs são montados
com peças reais (servos STS3215, peças impressas em 3D, ESP32 e bateria LiPo),
treinados com física propositalmente variada para resistir às diferenças entre
simulação e mundo real, e qualquer um deles pode ser exportado como peças STL,
lista de materiais, guia de montagem e arquivo de controle. Código e documentação
técnica em inglês; os dados de hardware em `docs/` estão em português.

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

Dependencies: `mujoco >= 3.6`, `numpy >= 1.23`, `cma >= 3.3`, `ribs >= 0.12`,
`matplotlib >= 3.6`. Optional extras:

```bash
pip install -e ".[video]"      # video recording (imageio + ffmpeg)
pip install -e ".[cad]"        # STL parts in the export (build123d)
pip install -e ".[gpu]"        # MJX backend (jax with CUDA 12)
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

# Train on randomised models (default 2 per terrain on cpu, 0 on gpu)
python experiments/train.py --variants 4

# Check the elite on 50 randomised models per terrain (PASS if mean loss < 30%)
python experiments/validate.py --n 50

# Export a robot to build: parts, bill of materials, assembly, control file
python experiments/export.py --type 4legs-0wheels-8servos

# Plot the archive heatmap and the progress curves
python experiments/plot_archive.py
python experiments/plot_archive.py --show
```

Training writes `checkpoints/archive.pkl` after every batch. `replay.py` reads
that file by default (`--archive` changes it). `plot_archive.py` saves
`archive_heatmap.png` and `archive_history.png` to `checkpoints/` (`--out`
changes the directory).

`--variants N` scores every candidate gait on N randomised copies of each
terrain (friction, mass, servo torque and stiffness, joint play, 0 to 2
control periods of command delay) and takes the mean, so gaits that depend on
exact simulator details lose. It costs about N times more per evaluation, is
CPU only, and is stored in the archive. `validate.py` runs the elite on
`--n` fresh variants (`--seed`, `--workers`) and prints nominal, mean and
worst distance, fall rate and relative loss per terrain.

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

A run on the `terrain` preset with 3 workers and the default 2 variants takes
roughly 30 to 40 min for the initial batch and 6 to 8 min for each later batch
(5 s episodes, 0.004 s physics step, 120 CMA-ES evaluations per child; see
`--seconds`, `--variants`, `--budget-init` and `--budget-child`). The archive is saved after
every batch; stopping mid-batch loses only that batch. Keep the machine awake
and the process alive when the terminal closes:

```bash
tmux new -s darwin
systemd-inhibit --what=sleep python experiments/train.py --terrains terrain --batches 40 --workers 3
# detach with Ctrl+b d, reattach with: tmux attach -t darwin
```

## Exporting a robot to build

`experiments/export.py --type <type>` writes `exports/<type>/`:

| File | Content |
|------|---------|
| `control.json` | the evolved gait for the ESP32 firmware, format in `docs/formato-controle.md` |
| `bom.md`, `bom.csv` | servos, printed parts with quantities and material, electronics, screws, tyres |
| `assembly.md` | servo id per joint and leg, mounting direction, calibration step |
| `parts/*.stl` | torso plate, links, servo brackets, feet, wheels (needs the `cad` extra) |
| `robot.xml` | the MuJoCo model, for inspection |

Servo dimensions in `darwinbot/parts.py` (horn and mounting holes, body size)
are placeholders marked "measure a real STS3215 and update": check them
against a real servo before printing.

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

## What counts as success (honestly)

| Level | What it means | Promised? |
|-------|---------------|-----------|
| **Minimum** | Diverse robots that walk or roll in simulation, built only from real parts | Yes |
| **Good** | An elite that passes `validate.py` (keeps most of its distance on randomised physics) and exports a coherent kit | Yes, with a full training run |
| **Moonshot** | A printed and assembled robot that walks with its evolved gait | No, it needs the hardware |

The value is in the pipeline being correct end to end, from genome to parts
list, not in any single robot.

## Known limitations

- Wheels were far ahead of legs before the switch to real parts (large
  wheels spinning at 15 rad/s). With servo-speed wheels of 3 to 6 cm the gap
  should shrink, but it has not been measured on a full run yet.
- No firmware yet: the ESP32 side (Phase 5 of `docs/plano-hardware.md`) waits
  for real hardware. The control file format it will read is already fixed.
- Randomised training (`--variants`) is CPU only.
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
│   ├── sim_gpu.py      # optional MJX batch simulator
│   ├── randomize.py    # randomised model variants (sim-to-real)
│   ├── fitness.py      # episode score and multi-terrain aggregate
│   ├── inner.py        # CMA-ES inner loop
│   ├── evolve.py       # MAP-Elites archive (pyribs) and outer loop
│   ├── parts.py        # printable parts (build123d)
│   ├── control_file.py # JSON control file for the firmware
│   └── export.py       # robot -> parts, BOM, assembly, control file
├── experiments/
│   ├── train.py        # run evolution
│   ├── replay.py       # viewer or video for an elite
│   ├── validate.py     # elite on randomised models (sim-to-real check)
│   ├── export.py       # export a robot to build
│   └── plot_archive.py # heatmap and progress plots
├── tests/              # pytest suite (GPU tests skip without jax)
├── docs/               # hardware data, control file format
├── checkpoints/        # archive.pkl and plots
├── exports/            # export.py output (ignored by git)
├── pyproject.toml
└── requirements.txt
```

## References and credits

- J.-B. Mouret and J. Clune, "Illuminating search spaces by mapping elites",
  arXiv:1504.04909, 2015 (MAP-Elites).
- N. Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772, 2016;
  implementation: [pycma](https://github.com/CMA-ES/pycma).
- E. Todorov, T. Erez and Y. Tassa, "MuJoCo: A physics engine for model-based
  control", IROS 2012; [MuJoCo](https://github.com/google-deepmind/mujoco) and MJX.
- B. Tjanaka et al., "pyribs: A Bare-Bones Python Library for Quality Diversity
  Optimization", GECCO 2023; [pyribs](https://github.com/icaros-usc/pyribs).
- [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini): STS3215 joint
  parameters and hardware recipe used as reference.
- [Revolve2](https://github.com/ci-group/revolve2) and RoboGen: modular,
  printable evolved robots, a design reference.
- [build123d](https://github.com/gumyr/build123d): parametric CAD for the parts.

## License

MIT, see [LICENSE](LICENSE). To cite this project, use the "Cite this
repository" button on GitHub (`CITATION.cff`).

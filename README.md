# darwinbot

[![tests](https://github.com/mpena2099/darwinbot/actions/workflows/tests.yml/badge.svg)](https://github.com/mpena2099/darwinbot/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Quality-diversity evolution of robot bodies and locomotion controllers, built
from real parts so the result can be printed and assembled.

Every robot is a tree of real modules (Karl Sims and Revolve2 style): a core
(torso plate, ESP32, LiPo) with servo joints, passive links, wheels and feet
hanging from its slots, all driven by Feetech STS3215 servos. Bodies are evolved
with a MAP-Elites archive laid out by behaviour and cost, while a central
pattern generator with two harmonics per joint and load feedback (Tegotae) is
tuned with CMA-ES for each candidate body. Any robot in the archive can be
exported as printable parts, a bill of materials and a control file for an ESP32.

## Resumo em português

O DarwinBot inventa robôs e ensina cada um a andar, numa simulação física
(MuJoCo). A evolução decide o corpo, uma árvore de módulos: quantas pernas, de que
tamanho, em que direção cada junta gira e se a perna termina numa roda. Para cada
corpo, um otimizador (CMA-ES) procura o ritmo de movimento das juntas, com dois
harmônicos e um retorno da carga dos servos, que leva o robô mais longe. Em vez de
um único vencedor, o programa guarda o melhor robô de cada comportamento (MAP-Elites
por tempo de contato com o chão, altura do tronco e número de servos), formando um
catálogo de soluções diferentes. Os robôs são montados com peças reais (servos
STS3215, peças impressas em 3D, ESP32 e bateria LiPo), treinados com física
propositalmente variada para resistir às diferenças entre simulação e mundo real,
e qualquer um deles pode ser exportado como peças STL, lista de materiais, guia de
montagem e arquivo de controle. Código e documentação técnica em inglês; os dados
de hardware em `docs/` estão em português.

## In one paragraph

The program invents robots and teaches each one to move. A robot is a tree of
modules; evolution decides how many limbs, how long, which way each joint turns
and whether a limb ends in a wheel. For every new body, an optimiser searches for
the rhythm of joint movements that carries it farthest forward in a few simulated
seconds. The run does not keep a single winner: it keeps the best robot for each
*behaviour* (crawling low with few servos, trotting tall with many, rolling...),
so at the end you can browse a catalogue of different solutions.

## How it works

Two nested loops:

1. **Outer loop, MAP-Elites over behaviour.** The archive is a pyribs
   CVTArchive with about 300 cells over three descriptors measured in the nominal
   flat-floor episode: `duty` (fraction of time the ground-touching tips are on
   the floor), `height` (mean torso height over the body reach) and `cost`
   (number of servos). Each cell keeps the best body found there, so a run ends
   with a map of specialists instead of a single design. Parents are drawn with
   probability proportional to their novelty (mean distance to the 10 nearest
   elites in descriptor space), which pushes the search towards sparse regions.
2. **Inner loop, CMA-ES over the controller.** For a fixed body, CMA-ES searches
   the controller genes that this body actually uses. A promising body is
   therefore judged by the best gait it can learn, not by a lucky random
   controller.

Children are mutated copies of elites. The child starts its inner loop from the
parent controller, so learned gaits are inherited along with the body
(Lamarckian inheritance). Controller genes are indexed by the stable slot id of
each module, so a mutation that keeps a module keeps its control. Only the genes
that affect the child are optimised; the rest stay in the genome and become
active again if a module returns.

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
machine is about 12x slower.

### Viewing the different robot types

The archive keeps one robot per **cell** of the behaviour map: the best one found
with that duty, height and number of servos. A new robot only replaces the one in
its own cell, and only if it scores higher, so slow crawlers survive next to
fast rollers.

A type is a short name derived from the cell, `<servos>servos-<height>-<gait>`
(for example `8servos-mid-trot`):

| Part | Values | Meaning |
|------|--------|---------|
| `servos` | 1 to 16 | number of servos, the cost of the robot |
| `height` | `low`, `mid`, `high` | mean torso height over body reach: below 0.4, below 0.8, above |
| `gait` | `crawl`, `walk`, `trot`, `belly`, `roll` | duty, the share of time the tips touch the ground: above 0.8, 0.5 to 0.8, 0.15 to 0.5, below 0.15 (sliding on the body); `roll` when wheels carry more than half of the contact time |

Several cells can share a name; the later ones get the cell index appended
(`12servos-low-walk-133`).

Step 1, see which types the run has found, best first:

```
$ python experiments/replay.py --list
type                       fitness  duty  height  servos
8servos-mid-trot              0.11  0.34    0.45       8
12servos-low-trot            -0.03  0.41    0.28      12
12servos-low-walk-133        -0.03  0.52    0.33      12
...
```

Step 2, open one in the 3D viewer (or add `--record name.mp4` for a video):

```bash
python experiments/replay.py --type 8servos-mid-trot
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
| `control.json` | the evolved controller for the ESP32 firmware (format 2), described in `docs/formato-controle.md` |
| `bom.md`, `bom.csv` | servos, printed parts with quantities and material, electronics, screws, tyres |
| `assembly.md` | one row per servo in layout order: id, node slot, parent chain in words (for example `core right-front > joint pitch > joint roll`), axis, link length, sign; mounting direction and calibration step |
| `parts/*.stl` | one file per distinct part: torso plate with a mount at every used core slot, a link per catalogue length, servo bracket, a wheel per radius, foot (needs the `cad` extra) |
| `robot.xml` | the MuJoCo model, for inspection |

The export walks the module tree: every joint and link needs one link of its
length, every servo one bracket, every wheel and foot (explicit, or the free tip
of a limb) one print. The bill of materials counts them from the tree.

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

A tree of modules (`core`, `joint`, `link`, `wheel`, `foot`). The genome is a
fixed vector of `MORPH_LENGTH = MAX_NODES * 9 = 180` genes: one block of 9 genes
per node slot, with `MAX_NODES = 20`. Block 0 is always the core; the block index
is the stable slot id of the node. A block is

| Gene | Meaning |
|------|---------|
| `active` | the module exists when > 0.5 |
| `parent` | slot id of the parent module |
| `slot` | attachment slot on the parent (core: 8 slots, front, back, left and right sides; joint and link: tip and two sides) |
| `rot` | roll of the attachment, 0, 90, 180 or 270 degrees |
| `type` | joint, link, wheel or foot |
| `axis` | joint turns about pitch, yaw or roll |
| `length` | catalogue link, 0.06, 0.08, 0.10 or 0.12 m |
| `wheel` | wheel radius 0.03 - 0.06 m |
| `flip` | joint axis reversed (how a mirrored copy keeps the same signal) |

For the core block, `length` and `wheel` hold the torso length (0.15 - 0.40 m)
and width (0.10 - 0.25 m). `decode()` accepts any vector and always returns a
body within `MAX_NODES`, `MAX_SERVOS = 16` and depth 5: blocks that are
inactive, unreachable, clash on a slot or break a limit are dropped. A joint or
link with no child ends in an implicit foot.

Mutation works on the decoded tree: change a parameter, change a module type,
add a module or small subtree, remove a subtree, duplicate a subtree mirrored on
the opposite slot, or copy it to another free slot. The initial population mixes
small random trees with seeds that reproduce mirrored leg pairs, with and
without wheels.

### Controller genome (`darwinbot/controller.py`)

Length: `CTRL_LENGTH = 1 + MAX_NODES * 6 = 121`, indexed by the same node slot ids.

| Index | Gene | Range | Meaning |
|-------|------|-------|---------|
| 0 | `frequency` | 0.5 - 3.0 Hz | global natural frequency, `omega = 2*pi*frequency` |

Per node slot `s`, with base `b = 1 + 6 * s`:

| Offset | Gene | Range | Meaning |
|--------|------|-------|---------|
| +0, +1 | `amp1`, `phase1` | 0 - 0.8 rad, 0 - 2*pi | first harmonic |
| +2 | `offset` | -0.6 - 0.6 rad | joint centre (for a wheel: signed speed, -5.45 - 5.45 rad/s) |
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
`(amp1 + 2 * amp2) * omega <= 5.45 rad/s`, the servo speed limit. The
simulation updates the servo targets at 50 Hz and holds them in between, like
the ESP32.

## Real parts

Every simulated robot is built from parts you can buy or print: Feetech
STS3215 servos at 7.4 V (1.91 N*m stall, 5.45 rad/s, run at 70% of the torque),
60 g per servo, an ESP32 + 2S LiPo + regulator payload of 0.13 kg in the torso,
printed links of 60, 80, 100 or 120 mm, and self-collision between limbs. The
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

## Results

A short run on a 4-core laptop (`--terrains terrain --batches 5 --workers 3`,
about 20,000 episodes in 2 hours) filled 50 of the ~300 cells:

| Gait | Types | Best fitness |
|------|-------|--------------|
| `roll` | 18 | 0.95 |
| `trot` | 20 | 0.39 |
| `belly` | 6 | 0.09 |
| `walk` | 3 | 0.01 |
| `crawl` | 3 | 0.02 |

Servo counts go from 1 to 16 and the bodies are often asymmetric: wheeled
quadrupeds, a tilted body that rolls on one side wheel, long-legged spiders.
25 types move forward (fitness above 0.05), 8 of them with legs carrying most
of the ground contact. The best of those, `13servos-low-trot`, loses 1% on
randomised physics in `validate.py` (PASS) and exports as 13 servos and 31
printed parts. Wheels still win on speed, by a factor of about 2.4. The filled-cell count was still rising linearly
when the run stopped, so longer runs should find more.

![Evolved body plans](docs/img/robots.png)

![Best wheel-less gait](docs/img/gait.gif)

![MAP-Elites archive](docs/img/archive_map.png)

![Training progress](docs/img/progress.png)

![Printable parts of the best robot](docs/img/parts.png)

## What counts as success (honestly)

| Level | What it means | Promised? |
|-------|---------------|-----------|
| **Minimum** | Diverse robots that walk or roll in simulation, built only from real parts | Yes |
| **Good** | An elite that passes `validate.py` (keeps most of its distance on randomised physics) and exports a coherent kit | Yes, with a full training run |
| **Moonshot** | A printed and assembled robot that walks with its evolved gait | No, it needs the hardware |

The value is in the pipeline being correct end to end, from genome to parts
list, not in any single robot.

## Known limitations

- Wheels still lead: the best rolling robot covers about 2.4 times the
  distance of the best legged one, and the top of the archive is mostly `roll`.
- No firmware yet: the ESP32 side waits for real hardware. The control file format it will read is already fixed.
- Randomised training (`--variants`) is CPU only.
- The only feedback is the servo load. The controller never sees the world, so
  tasks that need steering or reacting (finding a doorway, avoiding a wall) are
  out of scope.
- A body whose parts overlap at spawn is not buildable and scores `UNSTABLE_SCORE`
  without being simulated (common among random trees).

## Project structure

```
darwinbot/
├── darwinbot/
│   ├── morphology.py   # module tree, genome, decode, mutate
│   ├── controller.py   # two-harmonic Tegotae CPG, genome, active genes
│   ├── builder.py      # module tree + terrain -> MJCF
│   ├── terrain.py      # terrain snippets and presets
│   ├── sim.py          # Episode: compile once, replay controllers
│   ├── sim_gpu.py      # optional MJX batch simulator
│   ├── randomize.py    # randomised model variants (sim-to-real)
│   ├── fitness.py      # episode score and multi-terrain aggregate
│   ├── inner.py        # CMA-ES inner loop
│   ├── evolve.py       # behaviour archive (pyribs CVT), novelty selection, outer loop
│   ├── parts.py        # printable parts (build123d)
│   ├── control_file.py # JSON control file for the firmware
│   └── export.py       # robot -> parts, BOM, assembly, control file
├── experiments/
│   ├── train.py        # run evolution
│   ├── replay.py       # viewer or video for an elite
│   ├── validate.py     # elite on randomised models (sim-to-real check)
│   ├── export.py       # export a robot to build
│   └── plot_archive.py # behaviour scatter and progress plots
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

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE). To cite this project, use the "Cite this
repository" button on GitHub (`CITATION.cff`).

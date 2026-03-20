# darwinbot

Co-evolution of robot morphology and locomotion control using genetic algorithms.

Robots evolve both their **body structure** (limb attachment angles, segment lengths) and **hybrid locomotion** (oscillator-driven joints + motorised wheels), targeting environments built for humans — corridors, ramps, doorways.

Simulation runs on **MuJoCo** with evaluation managed by the **DEAP** evolutionary framework.

## Project structure

```
darwinbot/
├── darwinbot/
│   ├── genome.py          # genome encoding / decoding (6 limbs × 19 genes = 114 genes)
│   ├── mjcf_builder.py    # genome → MJCF XML for MuJoCo
│   ├── simulator.py       # physics simulation + CPG oscillator control
│   ├── fitness.py         # multi-scenario fitness function
│   ├── evolution.py       # DEAP evolutionary loop (SBX + Gaussian)
│   └── scenarios/
│       ├── registry.py    # central scenario registry + phase presets
│       ├── flat.py        # Phase 1 — baseline flat floor
│       ├── surfaces.py    # Phase 2 — wood / concrete / carpet zones
│       ├── obstacles.py   # Phase 3 — step, staircase, ramp
│       └── confined.py    # Phase 4 — small room with opening
├── experiments/
│   ├── train.py           # entry point: run evolution
│   ├── visualise.py       # replay best individual in MuJoCo viewer
│   └── plot_progress.py   # real-time fitness plots (matplotlib)
├── tests/
│   ├── test_genome.py     # genome encode/decode, CPG signal tests
│   ├── test_fitness.py    # scorer unit tests (no MuJoCo needed)
│   └── test_registry.py   # scenario registry + weight normalisation tests
├── checkpoints/           # saved populations (.pkl) + best_robot.xml
├── pyproject.toml
└── requirements.txt
```

## Install

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv && source venv/bin/activate

# Install the package and its dependencies in editable mode
pip install -e .
```

Dependencies: `mujoco >= 3.6`, `deap >= 1.4`, `numpy >= 1.23`, `matplotlib >= 3.6`.

## Run

```bash
# Phase 1 — flat floor (default: 40 individuals, 50 generations)
python experiments/train.py

# Custom parameters
python experiments/train.py --pop 80 --gen 200 --workers 6

# Reproducible run with a fixed seed
python experiments/train.py --seed 42

# Phase 2 — flat + varied surfaces
python experiments/train.py --scenarios phase2

# Phase 3 — flat + surfaces + obstacles
python experiments/train.py --scenarios phase3

# Phase 4 — all scenarios (flat + surfaces + obstacles + confined)
python experiments/train.py --scenarios phase4

# Resume from checkpoint
python experiments/train.py --resume checkpoints/gen_0050.pkl

# Visualise best individual in MuJoCo GUI
python experiments/visualise.py --checkpoint checkpoints/gen_0050.pkl
python experiments/visualise.py --checkpoint checkpoints/gen_0050.pkl --scenario obstacles --speed 0.3

# Real-time progress plots (run in a second terminal while training)
python experiments/plot_progress.py
```

After training completes, the best robot's MJCF XML is saved to `checkpoints/best_robot.xml`.

## Run tests

```bash
pip install pytest
pytest tests/ -v
```

## Genome structure

Each individual encodes `6 limbs × 19 genes = 114 floats`.
Each limb has **3 segments in series** (like an insect leg), with an optional wheel at the tip.

### Per-limb layout

| Offset | Gene | Description | Range |
|--------|------|-------------|-------|
| 0 | `attach_angle` | Attachment point on body ellipse (rad) | 0 – 2π |
| 1 | `angle` | Insertion direction from body (rad) | −π/2 – π/2 |
| 2 | `seg0.length` | Segment 0 length (m) | 0.06 – 0.25 |
| 3 | `seg0.width` | Segment 0 radius (m) | 0.02 – 0.08 |
| 4 | `seg0.osc_freq` | Oscillator frequency (Hz) | 0.3 – 3.0 |
| 5 | `seg0.osc_phase` | Oscillator phase (rad) | 0 – 2π |
| 6 | `seg0.osc_amp` | Oscillator amplitude (rad) | 0.1 – 1.0 |
| 7–11 | `seg1.*` | Segment 1 (same 5 genes) | — |
| 12–16 | `seg2.*` | Segment 2 (same 5 genes) | — |
| 17 | `has_wheel` | Wheel present | 0 or 1 |
| 18 | `wheel_rad` | Wheel radius (m) | 0.04 – 0.12 |

## Evolutionary strategy

- **Selection**: Tournament (size 3)
- **Crossover**: Simulated Binary Crossover (SBX), η = 20
- **Mutation**: Gaussian perturbation (σ = 5% of gene range, per-gene prob = 0.15)
- **Elitism**: Best individual from Hall of Fame is preserved each generation
- **Parallelism**: `multiprocessing.Pool` (one process per individual)
- **Checkpointing**: population saved every 10 generations to `checkpoints/`

## Scenarios

| Phase | Scenario | Description | Weight |
|-------|----------|-------------|--------|
| 1 | `flat` | Baseline flat floor | 1.0 |
| 2 | `surfaces` | Wood → concrete → carpet zones | 1.0 |
| 3 | `obstacles` | Step, staircase, ramp | 1.5 |
| 4 | `confined` | Small room with narrow opening | 1.0 |

Weights are **normalised** across active scenarios before evaluation.

## Fitness function

```
fitness = distance_x − 0.4 × lateral_dev − 0.001 × energy
```

- Multi-scenario: weighted sum of per-scenario scores (weights sum to 1.0)
- A fall returns a fixed penalty of −15
- Confined scenario awards milestones: entry (+2), back wall (+3), exit (+4)

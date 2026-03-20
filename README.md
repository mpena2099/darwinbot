# darwinbot

Co-evolution of robot morphology and locomotion control using genetic algorithms.

Robots evolve both their **body structure** (limb lengths, insertion angles) and **hybrid locomotion** (oscillator-driven joints + motorised wheels), targeting environments built for humans — corridors, ramps, doorways.

Simulation runs on **MuJoCo** with evaluation managed by the **DEAP** evolutionary framework.

## Project structure

```
darwinbot/
├── darwinbot/
│   ├── genome.py          # genome encoding / decoding (6 limbs × 9 genes)
│   ├── mjcf_builder.py    # genome → MJCF XML for MuJoCo
│   ├── simulator.py       # physics simulation + oscillator control
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
├── checkpoints/           # saved populations (.pkl)
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `mujoco >= 3.6`, `deap >= 1.4`, `numpy >= 1.23`, `matplotlib >= 3.6`.

## Run

```bash
# Phase 1 — flat floor (default: 40 individuals, 50 generations)
python experiments/train.py

# Custom parameters
python experiments/train.py --pop 80 --gen 200 --workers 6

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

# Real-time progress plots (run in a second terminal)
python experiments/plot_progress.py
```

## Genome structure

Each individual encodes `6 limbs × 9 genes = 54 floats`:

| Gene | Description | Range |
|------|-------------|-------|
| length | limb length (m) | 0.10 – 0.50 |
| width | limb radius (m) | 0.03 – 0.15 |
| angle | insertion angle (rad) | −π/2 – π/2 |
| has_wheel | wheel present | 0 or 1 |
| wheel_pos | position along limb | 0.3 – 0.9 |
| wheel_rad | wheel radius (m) | 0.04 – 0.12 |
| osc_freq | oscillator frequency (Hz) | 0.3 – 3.0 |
| osc_phase | oscillator phase (rad) | 0 – 2π |
| osc_amp | oscillator amplitude (rad) | 0.2 – 1.2 |

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

## Fitness function

```
fitness = distance_x − 0.4 × lateral_dev − 0.001 × energy
```

- Multi-scenario: weighted average of per-scenario scores
- A fall returns a fixed penalty of −15
- Confined scenario adds bonuses for entry (+2), reaching back wall (+3), and exit (+4)

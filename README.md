# darwinbot

Co-evolution of robot morphology and locomotion control using genetic algorithms.

Robots evolve both their **body structure** (limb lengths, insertion angles) and **hybrid locomotion** (oscillator-driven joints + motorised wheels), targeting environments built for humans — corridors, ramps, doorways.

## Project structure

```
darwinbot/
├── darwinbot/
│   ├── genome.py        # genome encoding / decoding
│   ├── urdf_builder.py  # genome → URDF for PyBullet
│   ├── simulator.py     # physics simulation + oscillator control
│   ├── fitness.py       # composite fitness function
│   └── evolution.py     # DEAP evolutionary loop
└── experiments/
    └── run_basic.py     # entry point
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Basic run (40 individuals, 50 generations)
python experiments/run_basic.py

# Custom parameters
python experiments/run_basic.py --pop 80 --gen 200 --workers 6

# Resume from checkpoint
python experiments/run_basic.py --resume checkpoints/gen_0050.pkl

# Visualise best individual in PyBullet GUI
python experiments/run_basic.py --gen 0 --visualise
```

## Genome structure

Each individual encodes `4 limbs × 9 genes = 36 floats`:

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

## Fitness function

```
fitness = distance_x − 0.4 × lateral_dev − 0.001 × energy
```

A fall returns a fixed penalty of −15.

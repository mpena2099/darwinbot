# About

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
│   ├── plot_archive.py # behaviour scatter and progress plots
│   └── lineage.py      # family tree of the elites
├── firmware/           # controller core in C++ for the ESP32, parity-tested
├── tests/              # pytest suite (GPU tests skip without jax)
├── docs/               # hardware data, control file format
├── checkpoints/        # archive.pkl and plots
├── exports/            # export.py output (ignored by git)
└── pyproject.toml
```

## Related work

| Project | What it does | Difference from DarwinBot |
|---|---|---|
| [RoboGen](https://github.com/lis-epfl/robogen) (EPFL) | Co-evolves bodies and controllers, outputs printable parts and Arduino code | Own ODE-based simulator, hobby micro servos, one winner per run |
| [Revolve2](https://github.com/ci-group/revolve2) (VU Amsterdam) | Research framework for modular robots, CPG brains, a runtime for physical robots | A library for researchers built around its lab's own modules; no part or BOM export |
| [DyRET](https://arxiv.org/abs/2003.13254) (Univ. of Oslo) | Open-hardware legged robot that evolves its leg lengths and gait in the real world | One fixed robot design, Dynamixel servos; does not invent bodies |
| [Nordmoen et al. 2021](https://arxiv.org/abs/2012.04375) | MAP-Elites on modular robots, stepping-stone analysis | A study, not a tool |
| [RoboMorph](https://arxiv.org/abs/2407.08626) | LLM-generated morphologies evaluated in simulation | Simulation only |

DarwinBot sits between them: a quality-diversity catalogue instead of a single
winner, and every robot in it exports as a kit of current, affordable parts
(STS3215 bus servos, printed links, ESP32) with a control file, from one
command.

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

Issues and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/mpena2099/darwinbot/blob/main/CONTRIBUTING.md).

## License

MIT, see [LICENSE](https://github.com/mpena2099/darwinbot/blob/main/LICENSE). To cite this project, use the "Cite this
repository" button on GitHub (`CITATION.cff`).

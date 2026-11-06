# Third-party notices

DarwinBot is MIT licensed (see `LICENSE`). It ships no third-party files; it
depends on the projects below and takes measures from the sources listed.

## Measures

- **Feetech STS3215 servo body and horn pattern** (`darwinbot/parts.py`):
  taken from the STEP model `STEP/SO100/STS3215_03a.step` of
  [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) by The Robot
  Studio (Apache-2.0), checked against the ST3215 STEP model and 2D drawing
  from [Waveshare](https://www.waveshare.com/wiki/ST3215_Servo). No file of
  either is included.
- **STS3215 dynamics** (`docs/hardware-dados.md`): compared with the SO-101
  MuJoCo model `Simulation/SO101/so101_new_calib.xml` of SO-ARM100
  (Apache-2.0).

## Dependencies

| Project | Used for | License |
|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | physics simulation | Apache-2.0 |
| [MuJoCo MJX](https://github.com/google-deepmind/mujoco) | GPU simulation (optional) | Apache-2.0 |
| [JAX](https://github.com/jax-ml/jax) | GPU backend (optional) | Apache-2.0 |
| [pyribs](https://github.com/icaros-usc/pyribs) | MAP-Elites archive | MIT |
| [pycma](https://github.com/CMA-ES/pycma) | CMA-ES | BSD-3-Clause |
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause |
| [Numba](https://numba.pydata.org) | compiled controller loop | BSD-2-Clause |
| [Matplotlib](https://matplotlib.org) | plots | PSF-based (Matplotlib license) |
| [build123d](https://github.com/gumyr/build123d) | printed parts, STL export (optional) | Apache-2.0 |
| [imageio](https://github.com/imageio/imageio) with ffmpeg | video (optional) | BSD-2-Clause; ffmpeg LGPL/GPL |
| [MkDocs](https://www.mkdocs.org) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) | documentation site | BSD-2-Clause, MIT |

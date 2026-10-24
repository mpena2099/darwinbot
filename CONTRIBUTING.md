# Contributing to DarwinBot

Thanks for your interest. DarwinBot is a personal hobby and portfolio
project, so reviews may take a while, but issues, ideas and pull requests are
welcome.

## Good contributions

- Bug reports with the command you ran, the full error and your OS and
  Python version.
- Measurements of a real Feetech STS3215 (horn hole pattern, mounting holes,
  body size, mass): `darwinbot/parts.py` still uses placeholders.
- Results from building or printing an exported robot, even partial ones.
- New terrains, behaviour descriptors or body modules that keep robots
  buildable from real parts.
- A GPU benchmark of the MJX backend on a modern NVIDIA card.

Before a large change, open an issue to discuss it.

## Development setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[video,cad]" pytest
pytest tests -q
```

GPU tests skip themselves unless `jax` and `mujoco-mjx` are installed
(`pip install -e ".[gpu]"`).

A quick end-to-end check:

```bash
python experiments/train.py --terrains flat --init 4 --batches 1 --batch-size 4 \
    --budget-init 10 --budget-child 6 --workers 1 --variants 0 --out /tmp/dbot
python experiments/replay.py --archive /tmp/dbot/archive.pkl --list
```

## Guidelines

- Keep code readable and close to the style of the surrounding module:
  a docstring header per module, short comments that explain why.
- Add or update tests for behaviour you change. CI runs `pytest` on Linux,
  Windows and macOS with Python 3.10 and 3.11.
- Physics or control changes in `darwinbot/sim.py` must be mirrored in
  `darwinbot/sim_gpu.py`; `tests/test_sim_gpu.py` compares both.
- If you change the genome or the physics in a way that makes old archives
  meaningless, bump `BODY_MODEL` in `darwinbot/evolve.py`.
- One topic per pull request, with a short description of what and why.

## License

By contributing you agree that your contributions are licensed under the
MIT License of this project.

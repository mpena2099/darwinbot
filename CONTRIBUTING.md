# Contributing to DarwinBot

Thanks for your interest. DarwinBot is a personal hobby and portfolio
project, so reviews may take a while, but issues, ideas and pull requests are
welcome.

## Good contributions

- Bug reports with the command you ran, the full error and your OS and
  Python version.
- A hardware component you own: a servo, battery, board, material or printer
  as one YAML file (see below).
- Measurements of a real Feetech STS3215 (horn hole pattern, mounting holes,
  body size, mass): the housing measures come from the SO-ARM100 STEP model,
  not from a caliper.
- Results from building or printing an exported robot, even partial ones.
- New terrains and tasks as plugins (`docs/plugins.md`), or body modules that
  keep robots buildable from real parts (open an issue first: a module still
  touches several files, see the roadmap).
- A GPU benchmark of the MJX backend on a modern NVIDIA card.

Before a large change, open an issue to discuss it.

## Contributing a hardware component

Every hardware number lives in one YAML file per component under
`darwinbot/profiles/` (`servos/`, `batteries/`, `electronics/`, `materials/`,
`printers/`), and a kit in `profiles/kits/` names one of each
(`docs/hardware-profiles.md`).

1. Copy the closest shipped file, rename it after the part
   (`servos/mg996r-6v.yaml`) and change the numbers. Put the source of every
   number in a comment: datasheet link, your own measurement (how), or an
   estimate (say so).
2. Write a kit that uses it and check it:
   `python tools/check_kit.py my-kit.yaml`. Fix every error; explain any
   warning in the pull request.
3. Run a short training with it (`train.py --profile my-kit.yaml`, the quick
   check below is enough) and say in the pull request what you saw.

A servo must keep the STS3215 housing layout (output axis along the body,
horns on both faces, same hole pattern); a part of another shape needs code
changes, so open an issue for it.

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
- Add or update tests for behaviour you change. CI runs `pytest` on Linux
  with Python 3.10 and 3.12. If you work on Windows or macOS, run the tests
  there before opening a pull request.
- Physics or control changes in `darwinbot/sim.py` must be mirrored in
  `darwinbot/sim_gpu.py` (`tests/test_sim_gpu.py` compares both) and in the
  firmware core when they touch control (`tests/test_firmware_parity.py`).
- If you change the genome or the physics incompatibly, bump `BODY_MODEL` in
  `darwinbot/evolve.py` and add a `MIGRATIONS` entry so old archives can be
  migrated (`experiments/migrate_archive.py`); leave it out only when no
  translation exists.
- One topic per pull request, with a short description of what and why.

## License

By contributing you agree that your contributions are licensed under the
MIT License of this project.

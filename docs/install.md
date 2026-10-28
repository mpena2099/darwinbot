# Install

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

## Tests

```bash
pip install pytest
pytest tests/ -q
```

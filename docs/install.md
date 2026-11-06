# Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
```

Dependencies: `mujoco >= 3.6`, `numpy >= 1.23`, `cma >= 3.3`, `ribs >= 0.12`,
`matplotlib >= 3.6`, `numba >= 0.58`. Optional extras:

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

## Shortcuts and Docker

The `Makefile` wraps the common commands (`make help` lists them):

```bash
make install
make test
make train BATCHES=30 WORKERS=3
make list ARCHIVE=checkpoints/steer/archive.pkl
make replay TYPE=12servos-low-belly
make export TYPE=12servos-low-belly
```

The `Dockerfile` builds a CPU image with the video and CAD extras, rendering
offscreen through EGL:

```bash
docker build -t darwinbot .
docker run --rm -v "$PWD/checkpoints:/app/checkpoints" darwinbot make train
```

"""train.py --deterministic: a batch's results are taken in submission order, so
two runs with the same seed and worker count give the same archive."""
import numpy as np

from darwinbot import evolve


def _run(out, **kw):
    return evolve.run(terrains="flat", n_init=3, n_batches=2, batch_size=3,
                      budget_init=4, budget_child=3, workers=2, seed=11,
                      episode_seconds=0.4, variants=0, out_dir=str(out), **kw)


def _same(a, b):
    assert a.stats() == b.stats()
    assert a.cells.keys() == b.cells.keys()
    for k in a.cells:
        assert a.cells[k].fitness == b.cells[k].fitness
        assert np.array_equal(a.cells[k].morph, b.cells[k].morph)
        assert np.array_equal(a.cells[k].ctrl, b.cells[k].ctrl)


def test_deterministic_runs_repeat(tmp_path):
    _same(_run(tmp_path / "a", deterministic=True),
          _run(tmp_path / "b", deterministic=True))


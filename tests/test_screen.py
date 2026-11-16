"""train.py --screen: a quick flat-ground score drops the worst changed children
before the inner loop."""
import numpy as np
import pytest

from darwinbot import evolve


def _run(out, **kw):
    return evolve.run(terrains="flat", n_init=3, n_batches=3, batch_size=4,
                      budget_init=4, budget_child=3, workers=2, seed=11,
                      episode_seconds=0.4, variants=0, out_dir=str(out), **kw)


def _same(a, b):
    assert a.stats() == b.stats()
    assert a.cells.keys() == b.cells.keys()
    for k in a.cells:
        assert a.cells[k].fitness == b.cells[k].fitness
        assert np.array_equal(a.cells[k].morph, b.cells[k].morph)
        assert np.array_equal(a.cells[k].ctrl, b.cells[k].ctrl)


def test_screen_off_changes_nothing(tmp_path):
    a, b = _run(tmp_path / "a", deterministic=True), _run(tmp_path / "b", deterministic=True, screen=1.0)
    _same(a, b)
    assert a.evals == b.evals


def test_screen_evaluates_fewer_bodies(tmp_path, capfd):
    full = _run(tmp_path / "a", deterministic=True)
    capfd.readouterr()
    part = _run(tmp_path / "b", deterministic=True, screen=0.25)
    out = capfd.readouterr().out
    assert part.search["screen"] == 0.25
    assert part.cells and part.evals < full.evals
    lines = [l for l in out.splitlines() if l.startswith("batch")]
    assert lines and all("screened=" in l for l in lines)
    assert any(not l.split("screened=")[1].startswith("0") for l in lines)


def test_screen_deterministic_repeats(tmp_path):
    _same(_run(tmp_path / "a", deterministic=True, screen=0.5),
          _run(tmp_path / "b", deterministic=True, screen=0.5))


def test_screen_keep_selection():
    origins = ["random", "same", "mutate", "graft", "mutate", "mutate"]
    scores = {2: 1.0, 3: 5.0, 4: 1.0, 5: -float("inf")}
    # 4 screened, keep ceil(0.5 * 4) = 2: the best (3), then the tie 2 vs 4 by order
    assert evolve.screen_keep(origins, scores, 0.5) == {0, 1, 3, 2}
    assert evolve.screen_keep(origins, scores, 0.01) == {0, 1, 3}   # at least one
    assert evolve.screen_keep(origins, scores, 1.0) == set(range(6))
    assert evolve.screen_keep(["same", "random"], {}, 0.1) == {0, 1}   # never dropped


def test_screen_invalid(tmp_path):
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            _run(tmp_path / "x", screen=bad)
    with pytest.raises(ValueError):
        _run(tmp_path / "y", screen=0.5, device="gpu")

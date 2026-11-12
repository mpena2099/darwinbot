"""
test_bench_inner.py
-------------------
inner.optimise_controller strategies (default path unchanged) and
tools/speed/bench_inner.py end to end on a tiny archive.
"""

import importlib.util
import os
import pickle
import sys

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo
from darwinbot.inner import optimise_controller
from darwinbot.terrain import get_terrains

_spec = importlib.util.spec_from_file_location(
    "bench_inner",
    os.path.join(os.path.dirname(__file__), "..", "tools", "speed", "bench_inner.py"))
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench      # the process pool pickles _run by module name
_spec.loader.exec_module(bench)

FLAT = get_terrains("flat")
SECONDS = 0.2
MORPH = mo.template(1)
GENES = mo.encode(MORPH)
X0 = np.random.default_rng(0).uniform(size=co.CTRL_LENGTH)


def _optimise(**kw):
    kw.setdefault("x0", X0)
    kw.setdefault("budget", 5)
    kw.setdefault("sigma0", 0.12)
    kw.setdefault("seed", 1)
    kw.setdefault("seconds", SECONDS)
    kw.setdefault("popsize", 4)
    return optimise_controller(MORPH, FLAT, **kw)


def test_none_strategy_is_the_default_call():
    a = _optimise()
    b = _optimise(strategy=None)
    assert a.fitness == b.fitness
    assert np.array_equal(a.ctrl, b.ctrl) and a.scores == b.scores


@pytest.mark.parametrize("strategy", ["diagonal", "focused", "diagonal+focused"])
def test_each_strategy_returns_a_valid_result(strategy):
    parent = mo.mutate(GENES, np.random.default_rng(2))
    res = _optimise(strategy=strategy, parent_genes=parent)
    assert np.isfinite(res.fitness)
    assert res.ctrl.shape == (co.CTRL_LENGTH,) and res.evals >= 1
    assert np.all((res.ctrl >= 0.0) & (res.ctrl <= 1.0))


def test_focused_without_parent_genes_is_the_baseline():
    a = _optimise()
    b = _optimise(strategy="focused", parent_genes=None)
    assert a.fitness == b.fitness and np.array_equal(a.ctrl, b.ctrl)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        _optimise(strategy="sideways")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return evolve.run(n_init=4, n_batches=0, batch_size=2, budget_init=4,
                      workers=1, episode_seconds=0.5,
                      out_dir=str(tmp_path_factory.mktemp("tiny")), variants=0)


def test_bench_inner_end_to_end(tiny, tmp_path, monkeypatch):
    archive = pickle.loads(pickle.dumps(tiny))
    archive.seconds = SECONDS
    path = tmp_path / "archive.pkl"
    with open(path, "wb") as f:
        pickle.dump(archive, f)
    out = tmp_path / "bench"
    monkeypatch.setattr(sys, "argv", [
        "bench_inner.py", "--archive", str(path), "--pairs", "2", "--seeds", "1",
        "--budget", "4", "--workers", "1", "--out", str(out), "--rng-seed", "1"])
    bench.main()
    assert (out / "runs.npz").exists() and (out / "summary.md").exists()
    text = (out / "summary.md").read_text()
    for s in bench.STRATEGIES:
        assert s in text
    z = np.load(out / "runs.npz")
    assert z["curves"].shape[0] == 2 * 1 * len(bench.STRATEGIES)
    assert set(z["strategy"]) == set(range(len(bench.STRATEGIES)))

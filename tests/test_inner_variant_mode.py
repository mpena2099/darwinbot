"""
test_inner_variant_mode.py
--------------------------
inner.optimise_controller's variant_mode: "all" stays the default, "first"
scores one variant per terrain in the loop and re-scores the best on the full
set once at the end.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import inner
from darwinbot import morphology as mo
from darwinbot.inner import optimise_controller
from darwinbot.terrain import get_terrains

FLAT, SECONDS, MORPH = get_terrains("flat"), 0.1, mo.template(1)
X0 = np.random.default_rng(0).uniform(size=co.CTRL_LENGTH)
KW = dict(x0=X0, budget=4, sigma0=0.12, seed=1, seconds=SECONDS, popsize=4)

_spec = importlib.util.spec_from_file_location("bench_inner_variant_mode", os.path.join(
    os.path.dirname(__file__), "..", "tools", "speed", "bench_inner.py"))
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench      # the process pool pickles _run by module name
_spec.loader.exec_module(bench)


def _optimise(**kw):
    return optimise_controller(MORPH, FLAT, **{**KW, **kw})


def test_default_mode_is_the_call_without_the_argument():
    a = _optimise()
    b = _optimise(variant_mode="all")
    assert a.fitness == b.fitness
    assert np.array_equal(a.ctrl, b.ctrl) and a.scores == b.scores
    with pytest.raises(ValueError, match="variant mode"):
        _optimise(variant_mode="half")


def test_first_mode_scores_one_variant_then_rescores_all(monkeypatch):
    calls, curve = [], []
    real = inner.evaluate

    def spy(episodes, ctrl, variants=None, task="forward", collect=None):
        calls.append([len(v) for v in variants] if variants else [0])
        return real(episodes, ctrl, variants, task, collect)

    monkeypatch.setattr(inner, "evaluate", spy)
    res = _optimise(variants=2, variant_mode="first", on_eval=curve.append)
    assert [c for c in calls if max(c) > 1] == [[2] * len(FLAT)]   # one full re-score
    assert calls[-1] == [2] * len(FLAT)
    assert all(c == [1] * len(FLAT) for c in calls[:-1])           # cheap loop
    assert len(curve) == res.evals + 1      # cheap points plus the re-score
    assert curve[-1] == pytest.approx(res.fitness)
    _, run_full = inner.cpu_evaluator(MORPH, FLAT, SECONDS, 0.0, 2, 1, "forward")
    assert run_full(res.ctrl)[0] == pytest.approx(res.fitness)


def test_bench_inner_accepts_the_variant_mode_flag():
    ap = bench._parser()
    base = ["--archive", "a.pkl", "--out", "bench"]
    assert ap.parse_args(base).variant_mode == "all"
    assert ap.parse_args(base + ["--variant-mode", "first"]).variant_mode == "first"
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--variant-mode", "half"])

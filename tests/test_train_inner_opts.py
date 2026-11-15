"""
test_train_inner_opts.py
------------------------
The inner-loop options of train.py (--inner-strategy, --variant-mode) reach
evolve.run and inner.optimise_controller; the defaults change nothing.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from darwinbot import evolve


def _run(out, variants=0, **kw):
    return evolve.run(terrains="flat", n_init=2, n_batches=1, batch_size=2,
                      budget_init=4, budget_child=3, workers=1, seed=7,
                      episode_seconds=0.4, variants=variants, out_dir=str(out), **kw)


def test_defaults_are_exactly_the_explicit_old_run(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b", inner_strategy=None, variant_mode="all")
    assert a.stats() == b.stats()
    ca, cb = a.cells, b.cells
    assert ca.keys() == cb.keys()
    for k in ca:
        assert ca[k].fitness == cb[k].fitness
        assert np.array_equal(ca[k].morph, cb[k].morph)
        assert np.array_equal(ca[k].ctrl, cb[k].ctrl)


def test_diagonal_focused_with_first_variants_fills_a_cell(tmp_path):
    archive = _run(tmp_path, inner_strategy="diagonal+focused",
                   variant_mode="first", variants=1)
    assert archive.cells


def test_unknown_inner_options_are_refused(tmp_path):
    with pytest.raises(ValueError, match="inner_strategy"):
        evolve.run(inner_strategy="diagonal+focused+extra", out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="variant_mode"):
        evolve.run(variant_mode="half", out_dir=str(tmp_path))


TRAIN_ARGS = ["train.py", "--init", "1", "--batches", "0", "--batch-size", "1",
              "--workers", "1", "--variants", "0", "--seconds", "1"]


def _train():
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "train.py")
    spec = importlib.util.spec_from_file_location("train_inner_opts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_passes_the_flags_to_evolve(tmp_path, monkeypatch):
    train = _train()
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        return evolve.Archive(terrains="flat")

    monkeypatch.setattr(train.evolve, "run", fake_run)
    base = TRAIN_ARGS + ["--out", str(tmp_path)]
    for extra, strategy, mode in (([], None, "all"),
                                  (["--inner-strategy", "diagonal+focused",
                                    "--variant-mode", "first"],
                                   "diagonal+focused", "first")):
        monkeypatch.setattr(sys, "argv", base + extra)
        train.main()
        assert (calls[-1]["inner_strategy"], calls[-1]["variant_mode"]) == (strategy, mode)
    assert len(calls) == 2


def test_train_rejects_an_unknown_strategy(tmp_path, monkeypatch):
    train = _train()
    monkeypatch.setattr(sys, "argv", TRAIN_ARGS + [
        "--out", str(tmp_path), "--inner-strategy", "bogus"])
    with pytest.raises(SystemExit):
        train.main()

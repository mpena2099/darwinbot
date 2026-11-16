"""
test_protect_new.py
-------------------
--protect-new: a child whose body changed a lot (a graft, PROTECT_SLOTS genome
slots, or another set of node slots) gets its inner-loop CMA-ES budget
multiplied, so a new body is not judged on a poorly tuned controller.
Default 1.0 turns it off and changes nothing; train.py refuses anything below 1.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from darwinbot import evolve


def _run(out, **kw):
    return evolve.run(terrains="flat", n_init=2, n_batches=1, batch_size=2,
                      budget_init=4, budget_child=3, workers=1, seed=7,
                      episode_seconds=0.4, variants=0, out_dir=str(out), **kw)


def test_default_is_off_and_bit_identical(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b", protect_new=1.0)
    assert a.stats() == b.stats()
    ca, cb = a.cells, b.cells
    assert ca.keys() == cb.keys()
    for k in ca:
        assert ca[k].fitness == cb[k].fitness
        assert np.array_equal(ca[k].morph, cb[k].morph)
        assert np.array_equal(ca[k].ctrl, cb[k].ctrl)


def test_protect_new_below_one_is_refused(tmp_path):
    with pytest.raises(ValueError, match="protect_new"):
        evolve.run(protect_new=0.5, out_dir=str(tmp_path))


def test_child_budget_doubles_a_big_change_and_keeps_a_small_one():
    big = [{"origin": "graft", "changed_slots": 1, "same_slots": True},
           {"origin": "mutate", "changed_slots": evolve.PROTECT_SLOTS,
            "same_slots": True},
           {"origin": "mutate", "changed_slots": 1, "same_slots": False}]
    small = [{"origin": "mutate", "changed_slots": 1, "same_slots": True},
             {"origin": "mutate", "changed_slots": 0, "same_slots": True},
             {"origin": "same", "changed_slots": 6, "same_slots": True},
             {"origin": "random", "changed_slots": None, "l2": None}]
    for m in big:
        assert evolve.child_budget(m, 120, 2.0) == 240
        assert evolve.child_budget(m, 120) == 120
    for m in small:
        assert evolve.child_budget(m, 120, 2.0) == 120
        assert evolve.child_budget(m, 120) == 120


def test_a_grafted_child_gets_the_bigger_budget_in_the_batch(tmp_path, monkeypatch):
    batches = []
    real = evolve.longest_first
    monkeypatch.setattr(evolve, "GRAFT_P", 1.0)
    monkeypatch.setattr(evolve, "longest_first",
                        lambda tasks: (batches.append([t[3] for t in tasks]),
                                       real(tasks))[1])
    evolve.run(terrains="flat", n_init=1, n_batches=2, batch_size=2,
               budget_init=4, budget_child=3, workers=1, seed=7,
               episode_seconds=0.4, variants=0, out_dir=str(tmp_path),
               protect_new=2.0)
    assert batches[0] == [4]                    # a random body keeps its budget
    for batch in batches[1:]:                   # every child here is a graft
        assert batch and batch == [6] * len(batch)


TRAIN_ARGS = ["train.py", "--init", "1", "--batches", "0", "--batch-size", "1",
              "--workers", "1", "--variants", "0", "--seconds", "1"]


def _train():
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "train.py")
    spec = importlib.util.spec_from_file_location("train_protect_new", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_passes_protect_new_and_refuses_below_one(tmp_path, monkeypatch):
    train = _train()
    calls = []
    monkeypatch.setattr(train.evolve, "run", lambda **kw: (calls.append(kw),
                                                          evolve.Archive(terrains="flat"))[1])
    base = TRAIN_ARGS + ["--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", base)
    train.main()
    assert calls[-1]["protect_new"] == 1.0
    monkeypatch.setattr(sys, "argv", base + ["--protect-new", "2"])
    train.main()
    assert calls[-1]["protect_new"] == 2.0
    monkeypatch.setattr(sys, "argv", base + ["--protect-new", "0.5"])
    with pytest.raises(SystemExit):
        train.main()

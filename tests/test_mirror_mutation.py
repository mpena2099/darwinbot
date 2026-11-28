"""
test_mirror_mutation.py
-----------------------
The mirrored mutation: a share of mutations (--mirror-p) copies the limb that
changed onto its mirror slot, so a symmetric body stays symmetric and a
mutation can build a symmetric one. The score never reads it, mirror_p=0 draws
no extra random number and changes nothing, and symmetry() counts the bodies
whose torso limbs have mirrored twins.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from darwinbot import evolve
from darwinbot.morphology import (decode, encode, mirror_slot, mutate, prune,
                                  random_morphology, symmetry, template,
                                  _drop_subtree)


def _one_limb() -> np.ndarray:
    """A body whose only torso limb sits on core slot 2 (mirror slot 5 free)."""
    body = decode(encode(template(2)))
    for s, c in list(body.child_map().get(0, {}).items()):
        if s != 2:
            _drop_subtree(body, c)
    return encode(prune(body))


def test_mirror_p_zero_is_the_old_mutation():
    bodies = [encode(template(2)), encode(template(1, wheels=True)),
              random_morphology(np.random.default_rng(3)),
              random_morphology(np.random.default_rng(4))]
    for genes in bodies:
        for seed in range(60):
            old = mutate(genes, np.random.default_rng(seed))
            new = mutate(genes, np.random.default_rng(seed), mirror_p=0.0)
            assert np.array_equal(old, new), seed


def test_mirror_p_fills_the_mirror_slot_with_a_copy():
    """One operator per mutation, so most trials only touch the limb: the copy
    that appears on the free mirror slot is its mirror and the body is
    symmetric."""
    genes = _one_limb()
    body = decode(genes)
    assert sorted(body.child_map().get(0, {})) == [2]
    assert symmetry(body) == 0.0            # one limb, no twin
    n = 60
    twinned = symmetric = 0
    for seed in range(n):
        child = decode(mutate(genes, np.random.default_rng(seed), n_ops=1,
                              mirror_p=1.0))
        if 5 not in child.child_map().get(0, {}):
            continue                        # the limb did not change this time
        twinned += 1
        if symmetry(child) == 1.0:
            symmetric += 1
    assert twinned >= n // 2
    assert symmetric >= int(0.8 * twinned)


def test_symmetry_of_a_template_and_of_an_asymmetric_change():
    body = template(2)                      # mirrored leg pairs on both sides
    assert symmetry(body) == 1.0
    body.nodes[min(i for i in body.nodes if i != 0)].length += 0.01
    assert symmetry(body) < 1.0
    assert symmetry(decode(encode(template(3)))) == 1.0


RUN = dict(terrains="flat", n_init=2, n_batches=1, batch_size=2,
           budget_init=4, budget_child=3, workers=1, seed=7,
           episode_seconds=0.4, variants=0)


def test_default_run_is_bit_identical(tmp_path):
    a = evolve.run(out_dir=str(tmp_path / "a"), **RUN)
    b = evolve.run(out_dir=str(tmp_path / "b"), mirror_p=0.0, **RUN)
    assert a.stats() == b.stats()
    assert a.cells.keys() == b.cells.keys()
    for k in a.cells:
        assert a.cells[k].fitness == b.cells[k].fitness
        assert np.array_equal(a.cells[k].morph, b.cells[k].morph)
        assert np.array_equal(a.cells[k].ctrl, b.cells[k].ctrl)


def test_mirror_p_outside_the_unit_interval_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mirror_p"):
        evolve.run(mirror_p=1.5, out_dir=str(tmp_path))


TRAIN_ARGS = ["train.py", "--init", "1", "--batches", "0", "--batch-size", "1",
              "--workers", "1", "--variants", "0", "--seconds", "1"]


def _train():
    path = os.path.join(os.path.dirname(__file__), "..", "experiments", "train.py")
    spec = importlib.util.spec_from_file_location("train_mirror_p", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_passes_mirror_p_and_refuses_out_of_range(tmp_path, monkeypatch):
    train = _train()
    calls = []
    monkeypatch.setattr(train.evolve, "run", lambda **kw: (calls.append(kw),
                                                          evolve.Archive(terrains="flat"))[1])
    base = TRAIN_ARGS + ["--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", base)
    train.main()
    assert calls[-1]["mirror_p"] == 0.0
    monkeypatch.setattr(sys, "argv", base + ["--mirror-p", "0.3"])
    train.main()
    assert calls[-1]["mirror_p"] == 0.3
    monkeypatch.setattr(sys, "argv", base + ["--mirror-p", "1.5"])
    with pytest.raises(SystemExit):
        train.main()

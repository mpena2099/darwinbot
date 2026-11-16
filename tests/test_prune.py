"""experiments/prune.py on a tiny archive: the pool gives the serial result."""

import sys

import numpy as np

sys.path.insert(0, "experiments")

from darwinbot import controller as ctrl   # noqa: E402
from darwinbot import morphology as mo   # noqa: E402
from darwinbot import evolve   # noqa: E402
import prune   # noqa: E402


def _archive():
    """One flat 1 s robot with a 4-joint template body and a random controller."""
    a = evolve.Archive(terrains="flat", seconds=1.0, variants=0, task="forward")
    rng = np.random.default_rng(0)
    e = evolve.Elite(mo.encode(mo.template(2)), rng.random(ctrl.CTRL_LENGTH), 0.1, [0.1], 0)
    assert a.insert(e)
    return a


def test_workers_1_and_2_prune_the_same_body():
    a = _archive()
    name = a.names()[next(iter(a.cells))]
    one = prune.prune_all(a, [name], budget=4, tol=0.02, seed=0, workers=1)
    two = prune.prune_all(a, [name], budget=4, tol=0.02, seed=0, workers=2)
    assert len(one) == len(two) == 1
    assert np.array_equal(one[0].morph, two[0].morph)
    assert np.array_equal(one[0].ctrl, two[0].ctrl)
    assert one[0].fitness == two[0].fitness
    assert one[0].scores == two[0].scores
    assert np.isfinite(one[0].fitness)
    assert len(one[0].scores) == 1       # one terrain: fitness and scores from one evaluation

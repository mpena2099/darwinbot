"""Golden tests: fixed bodies and controllers whose outputs are pinned.

They pin, for five seeded `random_morphology` bodies and the example robot
file `examples/robots/small-walker.yaml`: the sha256 of the MJCF text, the
metrics of an 8 s flat episode with a fixed controller, the behaviour vector,
the type name and (when build123d is installed) the sha256 of every file of the
export package. Any refactor that is meant to change nothing must leave them
untouched.

Regenerate `tests/golden/golden.json` with `python tools/update_golden.py` only
for an intended behaviour change (physics, score, genome, export), in the same
change as the `BODY_MODEL` bump or the A/B note that justifies it. Never to make a
refactor pass.
"""

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from darwinbot import builder, controller as co, morphology as mo, robot_file
from darwinbot.evolve import Elite, type_name
from darwinbot.inner import behaviour_vector
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden" / "golden.json"
SEEDS = range(5)
SECONDS = 8.0
CASES = [f"seed{s}" for s in SEEDS] + ["small-walker"]
ROUND = 9


def _case(name):
    """(genes, morphology, controller) of a named case."""
    if name == "small-walker":
        morph, _ = robot_file.load(ROOT / "examples" / "robots" / "small-walker.yaml")
        genes = mo.encode(morph)
        rng = np.random.default_rng(100)
    else:
        rng = np.random.default_rng(int(name[4:]))
        genes = mo.random_morphology(rng)
    return genes, mo.decode(genes), co.random_controller(rng)


def simulate(name):
    """Everything but the export package: the record pinned for one case."""
    genes, morph, ctrl = _case(name)
    flat = get_terrains("flat")[0]
    xml, _ = builder.build_mjcf(morph, flat.assets, flat.world)
    rec = {"xml": hashlib.sha256(xml.encode()).hexdigest()}
    ep = Episode(morph, flat, seconds=SECONDS)
    rec["buildable"] = bool(ep.buildable)
    if ep.buildable:
        m = ep.run(ctrl)
        d = dataclasses.asdict(m)
        rec["metrics"] = {k: round(float(d[k]), ROUND)
                          for k in ("dx", "progress", "energy", "fell", "duty", "height")
                          if k in d}
        beh = behaviour_vector(morph, m)
        rec["behaviour"] = [round(float(v), ROUND) for v in beh]
        rec["type"] = type_name(beh)
    return rec


def export_hashes(name):
    """sha256 of every file the export package of a case writes."""
    from darwinbot.export import export_robot
    genes, _, ctrl = _case(name)
    el = Elite(genes, ctrl, 1.0, [1.0], 0)
    out = tempfile.mkdtemp()
    export_robot(el, out, "golden", stl=True)
    files = {}
    for root, _, names in sorted(os.walk(out)):
        for f in sorted(names):
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                files[os.path.relpath(p, out)] = hashlib.sha256(fh.read()).hexdigest()
    return files


def _golden():
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name", CASES)
def test_simulation_golden(name):
    assert simulate(name) == _golden()[name]["sim"]


@pytest.mark.parametrize("name", CASES)
def test_export_golden(name):
    pytest.importorskip("build123d")
    want = _golden()[name]["export"]
    if want is None:
        pytest.skip("not buildable")
    assert export_hashes(name) == want

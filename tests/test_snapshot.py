"""Snapshots: a tiny archive, one robot, a 1 s recording."""

import pickle
import subprocess
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, "experiments")

from darwinbot import controller as ctrl   # noqa: E402
from darwinbot import morphology as mo    # noqa: E402
from darwinbot import evolve              # noqa: E402
import snapshot                           # noqa: E402

CRAWL = [0.9, 0.3, 4.0, 0.0, 1.0, 2.0]


def _archive(path, behaviour, fitness):
    a = evolve.Archive(terrains="flat", seconds=1.0, variants=0, task="forward")
    morph = mo.encode(mo.template(2))
    rng = np.random.default_rng(0)
    e = evolve.Elite(morph, rng.random(ctrl.CTRL_LENGTH), fitness, [fitness], 0,
                     np.array(behaviour))
    assert a.insert(e)
    with open(path, "wb") as f:
        pickle.dump(a, f)
    return a


def _read(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_snapshot_loads_without_darwinbot(tmp_path):
    src = tmp_path / "a.pkl"
    a = _archive(src, CRAWL, 0.05)
    out = tmp_path / "a.snap.pkl"
    snapshot.main(["--archive", str(src), "--seconds", "1", "--fps", "10"])
    assert out.exists()

    snap = _read(out)
    assert snap["format"] == 1
    assert snap["body_model"] == evolve.BODY_MODEL
    assert list(snap["robots"]) == [a.names()[min(a.cells)]]
    robot = next(iter(snap["robots"].values()))
    model = mujoco.MjModel.from_xml_string(robot["xml"])
    assert robot["qpos"].shape == (len(robot["qpos"]), model.nq)
    assert abs(len(robot["qpos"]) - round(1 * 10)) <= 1

    code = (
        "import pickle, sys\n"
        f"snap = pickle.load(open({str(out)!r}, 'rb'))\n"
        "assert 'darwinbot' not in sys.modules, "
        "sorted(m for m in sys.modules if 'darwinbot' in m)\n"
        "print(snap['format'], snap['body_model'], len(snap['robots']))\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_refuses_to_overwrite_an_existing_snapshot(tmp_path):
    src = tmp_path / "a.pkl"
    _archive(src, CRAWL, 0.05)
    out = tmp_path / "a.snap.pkl"
    out.write_bytes(b"mine")
    with pytest.raises(SystemExit, match="exists"):
        snapshot.main(["--archive", str(src), "--seconds", "1"])

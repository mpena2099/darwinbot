"""Snapshots: a tiny archive, one robot, a 1 s recording."""

import pickle
import subprocess
import sys

import mujoco
import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, "experiments")

from darwinbot import controller as ctrl   # noqa: E402
from darwinbot import morphology as mo    # noqa: E402
from darwinbot import evolve              # noqa: E402
import snapshot                           # noqa: E402

CRAWL = [0.9, 0.3, 4.0, 0.0, 1.0, 2.0]


def _archive(path, behaviour, fitness, seconds=1.0):
    a = evolve.Archive(terrains="flat", seconds=seconds, variants=0, task="forward")
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


def test_robots_from_two_archives_under_their_names(tmp_path):
    a_path, b_path = tmp_path / "a.pkl", tmp_path / "b.pkl"
    a = _archive(a_path, CRAWL, 0.05)
    b = _archive(b_path, [0.2, 0.8, 1.0, 0.5, 0.3, 0.9], 0.04)
    out = tmp_path / "both.snap.pkl"
    snapshot.main(["--robot", f"{a_path}:{a.names()[min(a.cells)].rsplit('-', 1)[0]}:flat-roller",
                   "--robot", f"{b_path}:{b.names()[min(b.cells)].rsplit('-', 1)[0]}:rough-belly",
                   "--seconds", "1", "--fps", "10", "--out", str(out)])
    snap = _read(out)
    assert sorted(snap["robots"]) == ["flat-roller", "rough-belly"]
    assert snap["task"] == "forward" and snap["seconds"] == 1.0
    for robot in snap["robots"].values():
        model = mujoco.MjModel.from_xml_string(robot["xml"])
        assert robot["qpos"].shape == (len(robot["qpos"]), model.nq)


def test_duplicate_robot_name_is_refused(tmp_path):
    a_path, b_path = tmp_path / "a.pkl", tmp_path / "b.pkl"
    a = _archive(a_path, CRAWL, 0.05)
    _archive(b_path, CRAWL, 0.04)
    kind = a.names()[min(a.cells)].rsplit("-", 1)[0]
    with pytest.raises(SystemExit, match="give each one its own NAME"):
        snapshot.main(["--robot", f"{a_path}:{kind}", "--robot", f"{b_path}:{kind}",
                       "--seconds", "1", "--out", str(tmp_path / "x.snap.pkl")])


def test_mismatched_seconds_are_refused(tmp_path):
    a_path, b_path = tmp_path / "a.pkl", tmp_path / "b.pkl"
    a = _archive(a_path, CRAWL, 0.05, seconds=1.0)
    _archive(b_path, CRAWL, 0.04, seconds=2.0)
    kind = a.names()[min(a.cells)].rsplit("-", 1)[0]
    with pytest.raises(SystemExit, match=r"seconds.*1\.0.*2\.0"):
        snapshot.main(["--robot", f"{a_path}:{kind}:one", "--robot", f"{b_path}:{kind}:two",
                       "--seconds", "1", "--out", str(tmp_path / "x.snap.pkl")])


def test_mismatched_body_model_is_refused():
    one = SimpleNamespace(body_model=24, seconds=1.0, task="forward")
    two = SimpleNamespace(body_model=23, seconds=1.0, task="forward")
    with pytest.raises(SystemExit, match=r"body_model.*24.*23"):
        snapshot.check_agree([("a.pkl", one), ("b.pkl", two)])

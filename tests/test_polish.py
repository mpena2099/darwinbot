"""The polish command on tiny archives: a couple of robots, a 1 s episode."""

import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, "experiments")

from darwinbot import controller as ctrl   # noqa: E402
from darwinbot import morphology as mo   # noqa: E402
from darwinbot import evolve   # noqa: E402
from darwinbot.inner import ControllerResult   # noqa: E402
import polish   # noqa: E402

CRAWL = [0.9, 0.3, 4.0, 0.0, 1.0, 2.0]
WALK = [0.6, 0.3, 4.0, 0.0, 1.0, 2.0]
TROT = [0.3, 0.3, 4.0, 0.0, 1.0, 2.0]


def _archive(path, robots):
    """Tiny flat archive: (behaviour, fitness) per robot, 4-joint body."""
    a = evolve.Archive(terrains="flat", seconds=1.0, variants=0, task="forward")
    morph = mo.encode(mo.template(2))
    rng = np.random.default_rng(0)
    for behaviour, fitness in robots:
        e = evolve.Elite(morph, rng.random(ctrl.CTRL_LENGTH), fitness, [fitness], 0,
                         np.array(behaviour))
        assert a.insert(e)
    with open(path, "wb") as f:
        pickle.dump(a, f)
    return a


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["polish.py", *argv])
    polish.main()


def _fake(monkeypatch, archive, behaviour, gain):
    """optimise_controller stub: same controller and a given behaviour, fit + gain."""
    table = {tuple(e.ctrl): e for e in archive.cells.values()}

    def fake(morph, terrains, x0=None, budget=0, **kw):
        old = table[tuple(x0)]
        return ControllerResult(np.array(x0), old.fitness + gain, list(old.scores), [],
                                5, np.array(behaviour if behaviour else old.behaviour))

    monkeypatch.setattr(polish, "optimise_controller", fake)


def _read(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_types_polishes_the_named_robot(tmp_path, monkeypatch):
    src = tmp_path / "a.pkl"
    a = _archive(src, [(CRAWL, 0.05)])
    out = tmp_path / "polished.pkl"
    _run(monkeypatch, "--archive", str(src), "--types", a.names()[min(a.cells)],
         "--out", str(out), "--budget", "4")
    b = _read(out)
    assert len(b.cells) == 1
    assert (b.terrains, b.seconds, b.variants, b.task, b.batch) == ("flat", 1.0, 0, "forward", 0)
    assert b.best().fitness >= 0.05


def test_per_gait_picks_the_best_of_each_gait(tmp_path, monkeypatch):
    src = tmp_path / "a.pkl"
    a = _archive(src, [(CRAWL, 0.10), (CRAWL, 0.80), (TROT, 0.50)])
    _fake(monkeypatch, a, None, gain=1.0)          # keep every behaviour in place
    out = tmp_path / "polished.pkl"
    _run(monkeypatch, "--archive", str(src), "--per-gait", "crawl,trot", "--out", str(out))
    b = _read(out)
    assert len(b.cells) == 2
    assert sorted(e.fitness for e in b.cells.values()) == [1.50, 1.80]


def test_refuses_to_overwrite_an_existing_output(tmp_path, monkeypatch):
    src = tmp_path / "a.pkl"
    a = _archive(src, [(CRAWL, 0.05)])
    out = tmp_path / "polished.pkl"
    out.write_bytes(b"mine")
    with pytest.raises(SystemExit, match="exists"):
        _run(monkeypatch, "--archive", str(src), "--types", a.names()[min(a.cells)],
             "--out", str(out))


def test_colliding_robots_go_to_a_second_archive(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.pkl"
    a = _archive(src, [(CRAWL, 0.05), (WALK, 0.06)])
    _fake(monkeypatch, a, [0.7, 0.4, 4.0, 0.0, 1.0, 2.0], gain=1.0)   # one shared cell
    names = ",".join(a.names()[c] for c in sorted(a.cells))
    out = tmp_path / "polished.pkl"
    _run(monkeypatch, "--archive", str(src), "--types", names, "--out", str(out))
    second = tmp_path / "polished-2.pkl"
    assert len(_read(out).cells) == 1 and len(_read(second).cells) == 1
    assert "polished-2.pkl" in capsys.readouterr().out


def test_a_worse_search_keeps_the_original_controller(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.pkl"
    a = _archive(src, [(CRAWL, 0.80)])
    _fake(monkeypatch, a, TROT, gain=-1.0)         # worse fitness and another cell
    out = tmp_path / "polished.pkl"
    _run(monkeypatch, "--archive", str(src), "--types", a.names()[min(a.cells)],
         "--out", str(out))
    b = _read(out)
    cell = min(a.cells)
    assert list(b.cells) == [cell]
    assert b.cells[cell].fitness == pytest.approx(0.80)
    assert np.array_equal(b.cells[cell].ctrl, a.cells[cell].ctrl)
    assert "kept the original" in capsys.readouterr().out

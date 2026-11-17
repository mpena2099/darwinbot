"""experiments/showcase.py: the stage plan, its commands and the argument errors."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import showcase   # noqa: E402


def test_stage_order():
    assert showcase.plan() == ["polish", "prune", "validate", "assembly", "snapshot",
                               "export"]


def test_skip_leaves_stages_out_but_keeps_the_order():
    assert showcase.plan(["polish", "prune"]) == ["validate", "assembly", "snapshot", "export"]
    assert showcase.plan(["assembly"]) == ["polish", "prune", "validate", "snapshot",
                                           "export"]


def test_polish_gets_workers_and_budget(tmp_path):
    argv = showcase.command("polish", "a.pkl", ["x", "y"], str(tmp_path),
                            workers=4, budget=7)
    assert argv[0] == sys.executable
    assert argv[1].endswith("experiments/polish.py")
    assert argv[argv.index("--types") + 1] == "x,y"
    assert argv[argv.index("--out") + 1] == str(tmp_path / "polished.pkl")
    assert argv[argv.index("--workers") + 1] == "4"
    assert argv[argv.index("--budget") + 1] == "7"


def test_prune_reads_the_polished_archive(tmp_path):
    argv = showcase.command("prune", str(tmp_path / "polished.pkl"), ["x"], str(tmp_path))
    assert argv[argv.index("--archive") + 1] == str(tmp_path / "polished.pkl")
    assert argv[argv.index("--out") + 1] == str(tmp_path / "pruned.pkl")


def test_validate_and_assembly_take_one_robot(tmp_path):
    for stage, script in (("validate", "experiments/validate.py"),
                          ("assembly", "tools/check_assembly.py")):
        argv = showcase.command(stage, "p.pkl", ["x"], str(tmp_path), workers=3)
        assert argv[1].endswith(script)
        assert argv[argv.index("--type") + 1] == "x"
        assert "--out" not in argv


def test_export_writes_into_exports_by_name(tmp_path):
    assert showcase.plan()[-1] == "export"
    argv = showcase.command("export", "f.pkl", ["7"], str(tmp_path), name="4servos-low-roll-7")
    assert argv[1].endswith("experiments/export.py")
    assert argv[argv.index("--type") + 1] == "7"
    assert argv[argv.index("--out") + 1] == str(tmp_path / "exports" / "4servos-low-roll-7")


def test_choose_prefers_pruned_then_polished_then_pruned():
    assert showcase.choose("PASS", "PASS") == "pruned"
    assert showcase.choose("FAIL", "PASS") == "pruned"
    assert showcase.choose("PASS", "FAIL") == "polished"
    assert showcase.choose("FAIL", "FAIL") == "pruned"
    assert showcase.choose("-", "-") == "pruned"          # validate skipped


def test_snapshot_writes_the_snapshot(tmp_path):
    argv = showcase.command("snapshot", "p.pkl", ["x", "y"], str(tmp_path))
    assert argv[argv.index("--out") + 1] == str(tmp_path / "showcase.snap.pkl")


def test_refuses_an_existing_out(tmp_path):
    out = tmp_path / "showcase"
    out.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        showcase.main(["--archive", "a.pkl", "--types", "x", "--out", str(out)])


def test_unknown_skip_stage(tmp_path):
    with pytest.raises(SystemExit, match="unknown stage"):
        showcase.main(["--archive", "a.pkl", "--types", "x",
                       "--out", str(tmp_path / "s"), "--skip", "polish,bake"])


def test_validate_gets_the_body_limit(tmp_path):
    argv = showcase.command("validate", "p.pkl", ["x"], str(tmp_path), max_body=0.3)
    assert argv[argv.index("--max-body") + 1] == "0.3"
    assert "--max-body" not in showcase.command("validate", "p.pkl", ["x"], str(tmp_path))
    assert "--max-body" not in showcase.command("prune", "p.pkl", ["x"], str(tmp_path),
                                                max_body=0.3)



def _archive_with_belly(path):
    """Two robots, one stored with its body on the ground 93% of the time."""
    import pickle
    import numpy as np
    from darwinbot import controller as ctl, evolve, morphology as mo
    from darwinbot.sim import Behaviour
    a = evolve.Archive(terrains="flat", seconds=1.0, variants=0, task="forward")
    rng = np.random.default_rng(0)
    for duty, bg, fit in ((0.3, 0.1, 0.5), (0.6, 0.93, 0.9)):   # other duty: another cell
        b = Behaviour(duty, 0.5, 4.0, 0.0, 2.0, 4.0, bg)
        a.insert(evolve.Elite(mo.encode(mo.template(2)), rng.random(ctl.CTRL_LENGTH),
                              fit, [fit], 0, b))
    with open(path, "wb") as f:
        pickle.dump(a, f)
    return {bg: a.names()[c] for c, bg in ((c, a.cells[c].behaviour.body_ground)
                                           for c in a.cells)}


def test_warns_about_a_belly_robot_before_the_chain(tmp_path, capsys):
    names = _archive_with_belly(tmp_path / "a.pkl")
    showcase.main(["--archive", str(tmp_path / "a.pkl"), "--types", ",".join(names.values()),
                   "--out", str(tmp_path / "s"), "--skip", ",".join(showcase.STAGES)])
    out = capsys.readouterr().out
    assert f"warning: {names[0.93]}: slides on its body 93% of the time" in out
    assert names[0.1] + ":" not in out


def test_strict_body_drops_the_belly_robot(tmp_path):
    import pickle
    names = _archive_with_belly(tmp_path / "a.pkl")
    showcase.main(["--archive", str(tmp_path / "a.pkl"),
                   "--types", ",".join(names.values()), "--out", str(tmp_path / "s"),
                   "--strict-body", "--skip", ",".join(showcase.STAGES)])
    with open(tmp_path / "s" / "final.pkl", "rb") as f:
        final = pickle.load(f)
    assert list(final.names().values()) == [names[0.1]]


def test_pruned_fail_with_polished_pass_ships_the_polished_version(tmp_path, monkeypatch):
    import dataclasses
    import pickle
    from darwinbot import evolve
    names = _archive_with_belly(tmp_path / "a.pkl")
    a = evolve.load_archive(tmp_path / "a.pkl")
    cell = next(c for c, e in a.cells.items() if e.behaviour.body_ground < 0.5)
    elite = a.cells[cell]
    uid = elite.uid

    def stage_archive(path, fitness):
        out = a.like()
        out.insert(dataclasses.replace(elite, fitness=fitness))
        with open(path, "wb") as f:
            pickle.dump(out, f)

    ran = []

    def fake_run(stage, argv, log=None):
        ran.append((stage, argv))
        out = Path(argv[argv.index("--out") + 1]).parent if "--out" in argv else None
        if stage == "polish":
            stage_archive(argv[argv.index("--out") + 1], 7.0)
        elif stage == "prune":
            stage_archive(argv[argv.index("--out") + 1], 6.0)
        elif stage.startswith("validate"):
            pruned = argv[argv.index("--archive") + 1].endswith("pruned.pkl")
            return "robot (n=5): " + ("FAIL" if pruned else "PASS")
        elif stage == "assembly":
            return "0 clash"
        return ""

    monkeypatch.setattr(showcase, "_run", fake_run)
    out = tmp_path / "s"
    showcase.main(["--archive", str(tmp_path / "a.pkl"), "--types", str(uid),
                   "--out", str(out)])
    with open(out / "final.pkl", "rb") as f:
        final = pickle.load(f)
    chosen = next(iter(final.cells.values()))
    assert chosen.uid == uid and chosen.fitness == 7.0          # the polished one
    summary = (out / "summary.txt").read_text()
    assert "PASS" in summary and "FAIL" in summary and "polished" in summary
    assert str(uid) in summary.split("\n")[1].split()
    exports = [argv for stage, argv in ran if stage == "export"]
    assert len(exports) == 1 and exports[0][exports[0].index("--archive") + 1].endswith(
        "final.pkl")
    assert {"validate polished %d" % uid, "validate pruned %d" % uid} <= {s for s, _ in ran}


def test_strict_body_with_only_belly_robots_stops(tmp_path):
    names = _archive_with_belly(tmp_path / "a.pkl")
    with pytest.raises(SystemExit, match="slides on its body"):
        showcase.main(["--archive", str(tmp_path / "a.pkl"), "--types", names[0.93],
                       "--out", str(tmp_path / "s"), "--strict-body"])

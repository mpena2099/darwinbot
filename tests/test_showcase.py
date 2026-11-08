"""experiments/showcase.py: the stage plan, its commands and the argument errors."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import showcase   # noqa: E402


def test_stage_order():
    assert showcase.plan() == ["polish", "prune", "validate", "assembly", "snapshot"]


def test_skip_leaves_stages_out_but_keeps_the_order():
    assert showcase.plan(["polish", "prune"]) == ["validate", "assembly", "snapshot"]
    assert showcase.plan(["assembly"]) == ["polish", "prune", "validate", "snapshot"]


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


def test_snapshot_is_the_last_stage_and_writes_the_snapshot(tmp_path):
    assert showcase.plan()[-1] == "snapshot"
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

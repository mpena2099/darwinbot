"""experiments/validate.py: the verdict criteria, each one failing alone."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from validate import Row, verdict  # noqa: E402


def row(terrain="flat", command="straight", nominal=1.0, mean=1.0, p5=1.0,
        worst=1.0, fall=0.0):
    return Row(terrain, command, nominal, mean, p5, worst, fall)


def test_all_criteria_pass():
    rows = [row(), row("rough", "left", 1.0, 0.95, 0.85, 0.7, 0.02)]
    assert verdict(rows) == (True, [])


def test_mean_loss_alone_fails():
    ok, reasons = verdict([row(mean=0.65, p5=0.6, worst=0.55)])
    assert not ok
    assert reasons == ["mean loss 35% (limit 30%)"]


def test_p5_loss_alone_fails():
    ok, reasons = verdict([row(mean=0.95, p5=0.3, worst=0.2)])
    assert not ok
    assert reasons == ["p5 loss >= 60%: flat/straight"]


def test_fall_rate_alone_fails():
    ok, reasons = verdict([row(fall=0.2)])
    assert not ok
    assert reasons == ["fall rate > 10%: flat/straight"]


def test_terrain_with_zero_nominal_progress_fails():
    rows = [row(), row("steps", "straight", nominal=0.0, mean=0.0, p5=0.0, worst=0.0)]
    ok, reasons = verdict(rows)
    assert not ok
    assert reasons == ["no progress on steps"]


def test_zero_nominal_rows_still_count_for_falls():
    ok, reasons = verdict([row(), row("steps", nominal=0.0, mean=0.0, p5=0.0,
                                      worst=0.0, fall=0.5)])
    assert not ok
    assert "no progress on steps" in reasons
    assert "fall rate > 10%: steps/straight" in reasons


def test_one_dead_command_on_a_moving_terrain_passes():
    rows = [row(), row(command="left", nominal=0.0, mean=0.0, p5=0.0, worst=0.0)]
    assert verdict(rows) == (True, [])


def test_empty_rows_fail():
    assert verdict([]) == (False, ["no rows"])

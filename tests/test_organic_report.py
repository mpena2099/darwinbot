"""
test_organic_report.py
----------------------
The export's organic check: the score line, and the warning with the polish
command when the hollow robot loses more than 10%.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import export as export_cli  # noqa: E402


def _report(monkeypatch, capsys, solid, hollow):
    monkeypatch.setattr(export_cli, "organic_check", lambda *a, **k: (solid, hollow))
    monkeypatch.setattr(export_cli.morpho, "decode", lambda genes: None)
    elite = SimpleNamespace(morph=np.zeros(1), ctrl=np.zeros(1))
    archive = SimpleNamespace(terrains="flat", seconds=8.0, task="forward",
                              payload=0.0, variants=0)
    export_cli.organic_report(elite, {}, archive, "a.pkl", "6servos-mid-trot")
    return capsys.readouterr().out


@pytest.mark.parametrize("hollow", [1.0, 0.95, 0.901])
def test_small_drop_prints_only_the_scores(monkeypatch, capsys, hollow):
    out = _report(monkeypatch, capsys, 1.0, hollow)
    assert out.startswith("organic check: solid 1.000")
    assert "warning" not in out


def test_drop_above_ten_percent_warns_with_the_polish_command(monkeypatch, capsys):
    out = _report(monkeypatch, capsys, 1.0, 0.85)
    assert "(-15%)" in out and "warning" in out
    assert ("python experiments/polish.py --archive a.pkl --types 6servos-mid-trot "
            "--organic --out 6servos-mid-trot-organic.pkl") in out


def test_no_warning_when_the_solid_robot_does_not_score(monkeypatch, capsys):
    out = _report(monkeypatch, capsys, -0.2, -0.5)
    assert "warning" not in out and "%" not in out

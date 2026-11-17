"""
test_check_kit.py
-----------------
The kit sanity check: plausible fields and units, the cross checks between
sections, and train.py refusing a bad --profile before it starts.
"""

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments"))

import check_kit                                   # noqa: E402
import train                                       # noqa: E402

from darwinbot import hardware                     # noqa: E402

DEFAULT = str(ROOT / "darwinbot" / "profiles" / "kits" / "default.yaml")


def _kit(tmp_path, mutate=None):
    """A one-file kit (all sections inline) with an optional change."""
    data = copy.deepcopy(hardware.default_profile())
    if mutate:
        mutate(data)
    path = tmp_path / "kit.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def _errors(path):
    return check_kit.check(path)[0]


def test_default_kit_has_no_errors():
    errors, warnings, report = check_kit.check(DEFAULT)
    assert errors == [] and warnings == []
    assert any("usable torque" in line for line in report)
    assert any("runtime" in line for line in report)
    assert any("bending" in line for line in report)


def test_servo_length_in_metres_errors(tmp_path):
    path = _kit(tmp_path, lambda d: d["servo"].update(length=0.045))
    errors = _errors(path)
    assert any("servo.length" in e and "metres" in e for e in errors)


def test_negative_mass_errors(tmp_path):
    path = _kit(tmp_path, lambda d: d["electronics"].update(mass=-0.1))
    assert any("electronics.mass" in e for e in _errors(path))


def test_stall_current_below_idle_errors(tmp_path):
    path = _kit(tmp_path, lambda d: d["servo"].update(current_stall=0.1))
    assert any("current_stall" in e and "current_idle" in e
               for e in _errors(path))


def test_unknown_field_errors(tmp_path):
    path = _kit(tmp_path, lambda d: d["servo"].update(extra_spin=1))
    assert any("unknown field 'servo.extra_spin'" in e
               for e in _errors(path))


def test_train_refuses_a_bad_profile(tmp_path):
    path = _kit(tmp_path, lambda d: d["servo"].update(length=0.045))
    with pytest.raises(SystemExit) as caught:
        train.check_kit_profile(path)
    assert "servo.length" in str(caught.value)


def test_example_pi_kit_is_clean():
    path = str(ROOT / "examples" / "kits" / "pi-zero-2w" / "pi-zero-2w.yaml")
    errors, warnings, _ = check_kit.check(path)
    assert errors == [] and warnings == []


def test_bad_holes_and_small_hole_diameter(tmp_path):
    path = _kit(tmp_path, lambda d: d["electronics"].update(holes=[1, 2, 3]))
    assert any("electronics.holes" in e for e in _errors(path))
    path = _kit(tmp_path, lambda d: d["electronics"].update(hole_d=2.0))
    assert any("hole_d" in w for w in check_kit.check(path)[1])


def test_old_inline_kit_without_board_keys_has_no_errors(tmp_path):
    def drop(d):
        for k in hardware.ELECTRONICS_DEFAULTS:
            del d["electronics"][k]
    assert _errors(_kit(tmp_path, drop)) == []

import copy
import os
import pickle
import subprocess
import sys

import pytest
import yaml

from darwinbot import builder, controller, evolve, hardware, morphology, power

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "darwinbot", "profiles", "sts3215-12v.yaml")


def test_default_values_are_the_old_literals():
    assert controller.SERVO_SPEED == 4.71
    assert controller.LOAD_FULL_SCALE == 0.7
    assert controller.SERVO_TORQUE == 2.94 * 0.7
    assert builder.SERVO_MASS == 0.060 + 0.015
    assert builder.PAYLOAD_MASS == 0.17
    assert builder.LINK_DENSITY == 1240.0
    assert builder.SHELL_DENSITY == 300.0
    assert (builder.JOINT_FRICTION, builder.JOINT_ARMATURE, builder.JOINT_KP,
            builder.WHEEL_KV) == (0.068, 0.027, 13.37, 2.0)
    assert builder.PLASTIC_FACTOR == 0.35
    assert morphology.SERVO_BOX == (0.045, 0.025, 0.035)
    assert morphology.SERVO_AXIS_OFFSET == 0.0125
    assert morphology.ELECTRONICS_BOX == (0.10, 0.06)
    assert morphology.PRINT_BODY_H == 0.0307
    assert (morphology.SERVO_STALL, morphology.PLA_STRENGTH,
            morphology.STRENGTH_SAFETY, morphology.BED) == (2.94, 50e6, 0.5, 0.22)
    assert (power.I_IDLE, power.I_STALL, power.ELECTRONICS_A) == (0.2, 2.7, 0.15)
    assert (power.BATTERY_AH, power.BATTERY_C, power.BATTERY_V) == (1.2, 20.0, 11.1)
    s = hardware.SERVO
    assert (s.length, s.width, s.height, s.body_height, s.z0, s.axis_x,
            s.clearance) == (45.4, 24.8, 35.0, 30.7, -15.5, 12.5, 0.4)
    assert s.tap_x == (5.0, -19.5) and s.tap_y == 10.2 and s.m2 == 2.4


def test_parts_and_export_read_the_profile():
    pytest.importorskip("build123d")
    from darwinbot import export, parts
    assert parts.SERVO_L == 45.4 and parts.SERVO_AXIS_X == 12.5
    assert parts.BED_MM == 220.0 and parts.HORN_T == 3.0
    assert export.SERVO_BODY == (45.4, 24.8, 30.7)
    assert export.CRADLE_SLACK == 0.4 + 2.5


def _run(env_path, code):
    env = dict(os.environ, DARWINBOT_PROFILE=env_path, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)


def _write(tmp_path, mutate):
    with open(DEFAULT) as f:
        data = yaml.safe_load(f)
    mutate(data)
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_changed_profile_changes_constants(tmp_path):
    def m(d):
        d["servo"]["speed"] = 5.5
        d["servo"]["mass"] = 0.07
    r = _run(_write(tmp_path, m),
             "from darwinbot import controller, builder;"
             "print(controller.SERVO_SPEED, builder.SERVO_MASS)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "5.5"
    assert float(r.stdout.split()[1]) == 0.07 + 0.015


def test_missing_and_unknown_keys_name_the_key(tmp_path):
    r = _run(_write(tmp_path, lambda d: d["servo"].pop("speed")),
             "import darwinbot.hardware")
    assert r.returncode != 0 and "servo.speed" in r.stderr
    r = _run(_write(tmp_path, lambda d: d["battery"].update(extra=1)),
             "import darwinbot.hardware")
    assert r.returncode != 0 and "battery.extra" in r.stderr
    r = _run(_write(tmp_path, lambda d: d.pop("printer")),
             "import darwinbot.hardware")
    assert r.returncode != 0 and "printer" in r.stderr


def test_archive_keeps_profile():
    a = evolve.Archive(terrains="flat")
    assert a.profile == hardware.PROFILE
    b = pickle.loads(pickle.dumps(a))
    assert b.profile == hardware.PROFILE


def test_old_archive_gets_default_profile():
    a = evolve.Archive(terrains="flat")
    state = dict(a.__dict__)
    state.pop("profile")
    b = evolve.Archive.__new__(evolve.Archive)
    b.__setstate__(state)
    assert b.profile == hardware.default_profile()


def test_merge_refuses_mixed_profiles():
    a = evolve.Archive(terrains="flat")
    b = evolve.Archive(terrains="flat")
    b.profile = copy.deepcopy(a.profile)
    b.profile["servo"]["speed"] = 9.0
    with pytest.raises(ValueError, match="profile"):
        evolve.merge([a, b])


def test_require_same_profile_is_a_no_op():
    hardware.require(copy.deepcopy(hardware.PROFILE))

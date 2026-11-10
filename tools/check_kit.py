"""
check_kit.py
------------
Sanity check of a hardware kit before a training run.

    python tools/check_kit.py my-kit.yaml

Every field must be a number in a plausible range for its unit, the sections
must agree, and the numbers the run would use are reported next to the default
kit. A kit may name components, point at files or inline whole sections
(docs/hardware-profiles.md). Errors stop `train.py --profile`; warnings do not.
"""

import argparse
import math
import os
import pathlib
import sys
from dataclasses import fields

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from darwinbot import hardware, morphology          # noqa: E402

N_SERVOS = 8             # a typical robot, for the runtime and mass estimates
AVG_LOAD = 0.3           # fraction of stall torque in the runtime estimate

# (unit, hard low, hard high, usual low, usual high). Outside the hard range is
# an error; outside the usual range is a warning. Bounds are inclusive.
SPEC = {
    "servo": {
        "speed":              ("rad/s", 0.001, 100.0, 0.5, 20.0),
        "stall_torque":       ("N*m", 0.001, 50.0, 0.05, 10.0),
        "load_full_scale":    ("fraction of stall", 0.001, 1.0, 0.1, 1.0),
        "mass":               ("kg", 0.001, 5.0, 0.005, 1.0),
        "cradle_mass_plate":  ("kg", 0.0, 5.0, 0.0, 0.5),
        "cradle_mass_link":   ("kg", 0.0, 5.0, 0.0, 0.5),
        "current_idle":       ("A", 0.0, 100.0, 0.001, 5.0),
        "current_stall":      ("A", 0.001, 500.0, 0.05, 50.0),
        "joint_friction":     ("N*m", 0.0, 100.0, 0.0, 10.0),
        "joint_armature":     ("kg*m^2", 0.0, 1.0, 0.0, 0.1),
        "joint_kp":           ("N*m/rad", 0.001, 1e6, 0.1, 1e4),
        "wheel_kv":           ("N*m/(rad/s)", 0.0, 1e4, 0.01, 1e3),
        "box":                ("m", 0.001, 1.0, 0.005, 0.5),
        "axis_offset":        ("m", 0.0, 0.5, 0.0, 0.1),
        "print_body_height":  ("m", 0.001, 0.5, 0.005, 0.2),
        "length":             ("mm", 1.0, 2000.0, 10.0, 300.0),
        "width":              ("mm", 0.5, 1000.0, 3.0, 150.0),
        "height":             ("mm", 1.0, 2000.0, 5.0, 300.0),
        "body_height":        ("mm", 0.5, 2000.0, 3.0, 200.0),
        "z0":                 ("mm", -2000.0, 2000.0, -500.0, 500.0),
        "axis_x":             ("mm", -2000.0, 2000.0, -200.0, 200.0),
        "clearance":          ("mm", 0.0, 100.0, 0.0, 5.0),
        "tap_x":              ("mm", -2000.0, 2000.0, -200.0, 200.0),
        "tap_y":              ("mm", 0.0, 1000.0, 0.0, 200.0),
        "m2":                 ("mm", 0.001, 100.0, 0.5, 20.0),
        "horn_bolt_circle":   ("mm", 1.0, 1000.0, 3.0, 100.0),
        "horn_hole_yaw":      ("degrees", -360.0, 360.0, 0.0, 90.0),
        "horn_n_holes":       ("holes", 1, 12, 2, 8),
        "horn_hole_d":        ("mm", 0.001, 100.0, 0.5, 20.0),
        "horn_bore_d":        ("mm", 0.001, 100.0, 0.5, 20.0),
        "horn_thickness":     ("mm", 0.001, 100.0, 0.5, 20.0),
    },
    "battery": {
        "capacity_ah":        ("Ah", 0.001, 100.0, 0.05, 20.0),
        "c_rating":           ("C", 0.001, 1000.0, 1.0, 100.0),
        "volts":              ("V", 0.1, 100.0, 3.0, 30.0),
    },
    "electronics": {
        "mass":               ("kg", 0.001, 10.0, 0.005, 5.0),
        "current_a":          ("A", 0.0, 100.0, 0.0, 10.0),
        "box":                ("m", 0.001, 1.0, 0.005, 0.5),
    },
    "material": {
        "link_density":       ("kg/m^3", 1.0, 30000.0, 100.0, 5000.0),
        "shell_density":      ("kg/m^3", 0.0, 30000.0, 10.0, 5000.0),
        "limb_density":       ("kg/m^3", 0.0, 30000.0, 10.0, 5000.0),
        "pla_strength":       ("Pa", 1e5, 1e10, 1e7, 2e8),
        "strength_safety":    ("fraction of PLA", 0.001, 1.0, 0.1, 1.0),
        "plastic_factor":     ("fraction of terrain", 0.0, 1.0, 0.01, 1.0),
    },
    "printer": {
        "bed":                ("m", 0.05, 1.0, 0.1, 0.6),
    },
}
TUPLE_LEN = {"servo": {"box": 3, "tap_x": 2}, "electronics": {"box": 2}}
INT_FIELDS = {("servo", "horn_n_holes")}


def _hint(unit, value) -> str:
    """A unit that looks swapped, the usual cause of an impossible number."""
    if unit == "mm" and 0 < abs(value) < 1:
        return " (a mm field, looks like metres)"
    if unit == "m" and abs(value) >= 2:
        return " (a m field, looks like millimetres)"
    return ""


def _check_field(section, name, value, label, errors, warnings):
    unit, lo, hi, usual_lo, usual_hi = SPEC[section][name]
    count = TUPLE_LEN.get(section, {}).get(name)
    if count is not None:
        if not isinstance(value, (list, tuple)) or len(value) != count:
            errors.append(f"{label} {section}.{name}: expected {count} numbers, "
                          f"got {value!r}")
            return
        values = value
    else:
        values = (value,)
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{label} {section}.{name}: {v!r} is not a number")
            continue
        if (section, name) in INT_FIELDS and int(v) != v:
            errors.append(f"{label} {section}.{name}: {v!r} is not a whole number")
            continue
        hint = _hint(unit, v)
        if not lo <= v <= hi:
            errors.append(f"{label} {section}.{name} = {v:g} {unit} outside "
                          f"{lo:g}..{hi:g}{hint}")
        elif not usual_lo <= v <= usual_hi:
            warnings.append(f"{label} {section}.{name} = {v:g} {unit} is unusual "
                            f"(expected {usual_lo:g}..{usual_hi:g}){hint}")


def _resolve(kit_path):
    """(kit data, {section: raw mapping}, {section: label}, errors) without
    raising, so every problem is reported in one run."""
    path = pathlib.Path(kit_path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        return None, {}, {}, [f"{path}: {e}"]
    if not isinstance(data, dict):
        return None, {}, {}, [f"{path}: the kit must be a mapping"]
    errors = [f"{path}: missing section '{name}'"
              for name in sorted(set(hardware.SECTIONS) - set(data))]
    errors += [f"{path}: unknown section '{name}'"
               for name in sorted(set(data) - set(hardware.SECTIONS))]
    parts, labels = {}, {}
    for name in hardware.SECTIONS:
        if name not in data:
            continue
        try:
            raw, label = hardware._component(name, data[name],
                                             path.resolve().parent, path)
        except (OSError, ValueError) as e:
            errors.append(str(e))
            continue
        parts[name], labels[name] = raw, label
    return data, parts, labels, errors


def _number(parts, section, name):
    value = (parts.get(section) or {}).get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _cross(parts, labels, errors, warnings):
    """Checks that need two fields, possibly from two sections."""
    idle = _number(parts, "servo", "current_idle")
    stall = _number(parts, "servo", "current_stall")
    if idle is not None and stall is not None and stall <= idle:
        errors.append(f"{labels['servo']} servo.current_stall = {stall:g} A must "
                      f"be above servo.current_idle = {idle:g} A")

    box = (parts.get("servo") or {}).get("box")
    if isinstance(box, (list, tuple)) and len(box) == 3:
        for axis, name in ((0, "length"), (1, "width")):
            mm = _number(parts, "servo", name)
            if mm is None or not isinstance(box[axis], (int, float)):
                continue
            expected = mm / 1000.0
            if expected > 0 and abs(box[axis] - expected) / expected > 0.2:
                warnings.append(
                    f"{labels['servo']} servo.box[{axis}] = {box[axis]:g} m is "
                    f"more than 20% from servo.{name} = {mm:g} mm")

    volts = _number(parts, "battery", "volts")
    name = labels.get("servo", "")
    if volts is not None and "12" in str(name) and not 9.0 <= volts <= 15.0:
        warnings.append(f"{labels['battery']} battery.volts = {volts:g} V is far "
                        f"from the 12 V the servo name suggests")


def _derived(parts, report):
    """The numbers a run will actually feel: torque, runtime, mass, stress."""
    stall = _number(parts, "servo", "stall_torque")
    scale = _number(parts, "servo", "load_full_scale")
    if stall is not None and scale is not None:
        report.append(f"usable torque: {stall * scale:.2f} N*m per servo "
                      f"({stall:.2f} x {scale:.2f})")

    idle = _number(parts, "servo", "current_idle")
    stall_a = _number(parts, "servo", "current_stall")
    board = _number(parts, "electronics", "current_a")
    capacity = _number(parts, "battery", "capacity_ah")
    if None not in (idle, stall_a, board, capacity) and capacity > 0:
        idle_a = N_SERVOS * idle + board
        avg = idle + AVG_LOAD * (stall_a - idle)
        avg_a = N_SERVOS * avg + board
        if idle_a > 0 and avg_a > 0:
            report.append(
                f"runtime on {capacity * 1000:.0f} mAh: {capacity * 60 / idle_a:.0f} min "
                f"at idle, {capacity * 60 / avg_a:.0f} min at {AVG_LOAD:.0%} stall "
                f"({N_SERVOS} servos, {board:.2f} A electronics)")

    mass = _number(parts, "servo", "mass")
    cradle = _number(parts, "servo", "cradle_mass_plate")   # the heavier cradle: an upper bound
    board_mass = _number(parts, "electronics", "mass")
    if None not in (mass, cradle, board_mass):
        total = N_SERVOS * (mass + cradle) + board_mass
        report.append(f"{N_SERVOS}-servo mass: {total:.2f} kg "
                      f"({N_SERVOS} x {mass:.3f} kg + {N_SERVOS} x {cradle:.3f} kg "
                      f"cradles on plates + {board_mass:.3f} kg electronics incl. pack)")

    pla = _number(parts, "material", "pla_strength")
    safety = _number(parts, "material", "strength_safety")
    if stall is None or pla is None or safety is None:
        return
    limit = pla * safety
    longest = morphology.LENGTH_RANGE[1]
    w_max = morphology.LINK_WIDTH_RANGE[1]
    t_max = morphology.LINK_THICK_RANGE[1]
    # Same section stress as morphology.violations: the stall torque over
    # width * thickness^2 / 6.
    stress = (stall / (w_max * t_max ** 2 / 6.0) / 1e6
              if w_max and t_max else math.inf)
    report.append(
        f"bending: longest link {longest * 1000:.0f} mm -> {stress:.1f} MPa at the "
        f"{w_max * 1000:.0f} x {t_max * 1000:.1f} mm section limit "
        f"({limit / 1e6:.1f} MPa allowed)")


def check(kit_path):
    """(errors, warnings, report) of a kit file; never raises for bad values."""
    _, parts, labels, errors = _resolve(kit_path)
    warnings = []
    if not parts and errors:
        return errors, warnings, []
    for section, cls in hardware.SECTIONS.items():
        if section not in parts:
            continue
        raw, label = parts[section], labels[section]
        if not isinstance(raw, dict):
            errors.append(f"{label} {section}: must be a mapping")
            continue
        keys = {f.name for f in fields(cls)}
        for name in sorted(keys - set(raw)):
            errors.append(f"{label} {section}: missing field '{section}.{name}'")
        for name in sorted(set(raw) - keys):
            errors.append(f"{label} {section}: unknown field '{section}.{name}'")
        for name, value in raw.items():
            if name in keys and name in SPEC[section]:
                _check_field(section, name, value, label, errors, warnings)
    _cross(parts, labels, errors, warnings)
    report = []
    _derived(parts, report)
    return errors, warnings, report


def _plain(value):
    return list(value) if isinstance(value, (list, tuple)) else value


def _diff(kit_path):
    """The fields that differ from the default kit, old -> new."""
    _, parts, _, _ = _resolve(kit_path)
    base = hardware.default_profile()
    out = []
    for section in hardware.SECTIONS:
        current = parts.get(section)
        if not isinstance(current, dict):
            continue
        for name, old in base[section].items():
            if name in current and _plain(current[name]) != _plain(old):
                out.append(f"{section}.{name}: {_plain(old)} -> "
                           f"{_plain(current[name])}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("kit", help="kit YAML (docs/hardware-profiles.md)")
    a = p.parse_args(argv)
    errors, warnings, report = check(a.kit)
    for message in errors:
        print(f"error: {message}")
    for message in warnings:
        print(f"warning: {message}")
    diff = _diff(a.kit)
    if diff:
        print("diff from the default kit:")
        for line in diff:
            print(f"  {line}")
    if report:
        print("derived numbers:")
        for line in report:
            print(f"  {line}")
    print(f"{a.kit}: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

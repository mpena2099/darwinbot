"""
hardware.py
-----------
One hardware profile, loaded once at import, that every module reads instead
of keeping its own literals: the servo, the battery, the electronics, the
printed material and the printer. The profile is a YAML file, the default
`darwinbot/profiles/sts3215-12v.yaml` or the path in `DARWINBOT_PROFILE`.

The values are exposed twice: as frozen dataclasses (`hardware.SERVO.speed`,
`hardware.BATTERY.volts`, ...) for the code, and as the plain `PROFILE` dict
that an archive stores, so a checkpoint knows the hardware it was trained on.
`require(profile)` switches the whole process to another profile by re-running
the same command with `DARWINBOT_PROFILE` set; the experiment scripts call it
right after loading an archive.
"""

import os
import sys
import tempfile
from dataclasses import dataclass, fields
from importlib import resources

import yaml

DEFAULT_NAME = "sts3215-12v"
ENV = "DARWINBOT_PROFILE"


@dataclass(frozen=True)
class Servo:
    speed: float
    stall_torque: float
    load_full_scale: float
    mass: float
    bracket_mass: float
    current_idle: float
    current_stall: float
    joint_friction: float
    joint_armature: float
    joint_kp: float
    wheel_kv: float
    box: tuple
    axis_offset: float
    print_body_height: float
    length: float
    width: float
    height: float
    body_height: float
    z0: float
    axis_x: float
    clearance: float
    tap_x: tuple
    tap_y: float
    m2: float
    horn_bolt_circle: float
    horn_hole_yaw: float
    horn_n_holes: int
    horn_hole_d: float
    horn_bore_d: float
    horn_thickness: float


@dataclass(frozen=True)
class Battery:
    capacity_ah: float
    c_rating: float
    volts: float


@dataclass(frozen=True)
class Electronics:
    mass: float
    current_a: float
    box: tuple


@dataclass(frozen=True)
class Material:
    link_density: float
    shell_density: float
    limb_density: float
    pla_strength: float
    strength_safety: float
    plastic_factor: float


@dataclass(frozen=True)
class Printer:
    bed: float


SECTIONS = {"servo": Servo, "battery": Battery, "electronics": Electronics,
            "material": Material, "printer": Printer}
_TUPLES = {"box", "tap_x"}


def _read(source):
    """(text, label) of a profile path or an importlib resource."""
    if hasattr(source, "read_text"):
        return source.read_text(encoding="utf-8"), str(source)
    with open(source, "r", encoding="utf-8") as f:
        return f.read(), str(source)


def _section(name, data, cls):
    if not isinstance(data, dict):
        raise ValueError(f"hardware profile: '{name}' must be a mapping")
    keys = {f.name for f in fields(cls)}
    missing = sorted(keys - set(data))
    unknown = sorted(set(data) - keys)
    if missing:
        raise ValueError(f"hardware profile: missing key '{name}.{missing[0]}'")
    if unknown:
        raise ValueError(f"hardware profile: unknown key '{name}.{unknown[0]}'")
    values = {}
    for f in fields(cls):
        value = data[f.name]
        values[f.name] = tuple(value) if f.name in _TUPLES else value
    return cls(**values)


def _parse(source):
    """(plain dict, {section: dataclass}) of a profile, validated."""
    text, label = _read(source)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"hardware profile {label}: the file must be a mapping")
    missing = sorted(set(SECTIONS) - set(data))
    unknown = sorted(set(data) - set(SECTIONS))
    if missing:
        raise ValueError(f"hardware profile {label}: missing key '{missing[0]}'")
    if unknown:
        raise ValueError(f"hardware profile {label}: unknown key '{unknown[0]}'")
    return data, {name: _section(name, data[name], cls)
                  for name, cls in SECTIONS.items()}


def load(path):
    """Read and validate a profile file, returning its plain dict."""
    return _parse(path)[0]


def default_profile():
    """The plain dict of the shipped default profile, whatever profile is
    active: what an archive from before profiles was trained on."""
    return _parse(resources.files("darwinbot").joinpath(
        "profiles", f"{DEFAULT_NAME}.yaml"))[0]


def require(profile):
    """Make `profile` the active one. When it differs from the profile loaded
    at import, write it to a temporary YAML and re-exec this command with
    DARWINBOT_PROFILE set, so every module reads the same values."""
    if profile == PROFILE:
        return
    fd, path = tempfile.mkstemp(prefix="darwinbot-profile-", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False)
    os.environ[ENV] = path
    print(f"[darwinbot] hardware profile differs: restarting with {path}", flush=True)
    os.execv(sys.executable, [sys.executable, *sys.argv])


_path = os.environ.get(ENV)
if _path:
    _source = _path
    NAME = os.path.splitext(os.path.basename(_path))[0]
else:
    _source = resources.files("darwinbot").joinpath(
        "profiles", f"{DEFAULT_NAME}.yaml")
    NAME = DEFAULT_NAME
PROFILE, _parts = _parse(_source)
SERVO = _parts["servo"]
BATTERY = _parts["battery"]
ELECTRONICS = _parts["electronics"]
MATERIAL = _parts["material"]
PRINTER = _parts["printer"]

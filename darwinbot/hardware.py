"""
hardware.py
-----------
One hardware kit, loaded once at import, that every module reads instead
of keeping its own literals: the servo, the battery, the electronics, the
printed material and the printer. A kit is a YAML file that names one
component per section (`servo: sts3215-12v`, a path, or an inline mapping);
the default is `darwinbot/profiles/kits/default.yaml`, or the file in
`DARWINBOT_PROFILE`. Components are YAML files in `darwinbot/profiles/<section
plural>/`, or in a folder of that name next to the kit.

The values are exposed twice: as frozen dataclasses (`hardware.SERVO.speed`,
`hardware.BATTERY.volts`, ...) for the code, and as the plain `PROFILE` dict
that an archive stores, so a checkpoint knows the hardware it was trained on.
`require(profile)` switches the whole process to another profile by re-running
the same command with `DARWINBOT_PROFILE` set; the experiment scripts call it
right after loading an archive.
"""

import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass, fields
from importlib import resources

import yaml

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


PLURALS = {"servo": "servos", "battery": "batteries", "electronics": "electronics",
           "material": "materials", "printer": "printers"}
DEFAULT_KIT = "default"


def _shipped():
    return resources.files("darwinbot").joinpath("profiles")


def _yaml(source):
    """(data, label) of a YAML path or importlib resource."""
    label = str(source)
    if hasattr(source, "read_text"):
        text = source.read_text(encoding="utf-8")
    else:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()
    return yaml.safe_load(text), label


def _section(name, data, cls, label):
    if not isinstance(data, dict):
        raise ValueError(f"hardware profile {label}: '{name}' must be a mapping")
    keys = {f.name for f in fields(cls)}
    missing = sorted(keys - set(data))
    unknown = sorted(set(data) - keys)
    if missing:
        raise ValueError(f"hardware profile {label}: missing key '{name}.{missing[0]}'")
    if unknown:
        raise ValueError(f"hardware profile {label}: unknown key '{name}.{unknown[0]}'")
    values = {}
    for f in fields(cls):
        value = data[f.name]
        values[f.name] = tuple(value) if f.name in _TUPLES else value
    return cls(**values)


def _component(section, entry, kit_dir, label):
    """The mapping of one kit entry: inline, a path to a YAML file relative to
    the kit, or a component name found next to the kit and then in the
    shipped profiles."""
    if isinstance(entry, dict):
        return entry, label
    if not isinstance(entry, str):
        raise ValueError(f"hardware profile {label}: '{section}' must be a "
                         "component name, a path or a mapping")
    if "/" in entry or entry.endswith((".yaml", ".yml")):
        path = kit_dir.joinpath(entry)
        if not path.is_file():
            raise ValueError(f"hardware profile {label}: {section} file "
                             f"'{entry}' not found ({path})")
    else:
        folder = PLURALS[section]
        candidates = [kit_dir.joinpath(folder, entry + ".yaml"),
                      _shipped().joinpath(folder, entry + ".yaml")]
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            raise ValueError(
                f"hardware profile {label}: no {section} component '{entry}' "
                f"(looked in {candidates[0]} and {candidates[1]})")
    data, clabel = _yaml(path)
    return data, clabel


def _parse(source, kit_dir=None):
    """(plain dict, {section: dataclass}) of a kit file, validated. A kit
    entry is a component name, a path or an inline mapping; the plain dict is
    the resolved {section: {key: value}}."""
    data, label = _yaml(source)
    if kit_dir is None:
        kit_dir = pathlib.Path(source).resolve().parent
    if not isinstance(data, dict):
        raise ValueError(f"hardware profile {label}: the file must be a mapping")
    missing = sorted(set(SECTIONS) - set(data))
    unknown = sorted(set(data) - set(SECTIONS))
    if missing:
        raise ValueError(f"hardware profile {label}: missing key '{missing[0]}'")
    if unknown:
        raise ValueError(f"hardware profile {label}: unknown key '{unknown[0]}'")
    plain, parts = {}, {}
    for name, cls in SECTIONS.items():
        plain[name], clabel = _component(name, data[name], kit_dir, label)
        parts[name] = _section(name, plain[name], cls, clabel)
    return plain, parts


def load(path):
    """Read and validate a kit file, returning its resolved plain dict."""
    return _parse(path)[0]


def default_profile():
    """The plain dict of the shipped default kit, whatever profile is active:
    what an archive from before profiles was trained on."""
    return _parse(_shipped().joinpath("kits", f"{DEFAULT_KIT}.yaml"),
                  _shipped().joinpath("kits"))[0]


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
    _source = _shipped().joinpath("kits", f"{DEFAULT_KIT}.yaml")
    NAME = DEFAULT_KIT
PROFILE, _parts = _parse(_source, None if _path else _shipped().joinpath("kits"))
SERVO = _parts["servo"]
BATTERY = _parts["battery"]
ELECTRONICS = _parts["electronics"]
MATERIAL = _parts["material"]
PRINTER = _parts["printer"]

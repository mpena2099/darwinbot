"""
control_file.py
---------------
Writes and reads the JSON control file of docs/formato-controle.md (format 3),
the hand-off from the evolved Tegotae CPG to the ESP32 firmware.

Format 3 adds the two steering genes to format 2. Files still carry the
unsteered amplitudes and offsets; the firmware applies the steering command and
only then the speed clip.

No dependency on morphology or builder: it takes the actuator layout, the
controller parameters and the joint limits as plain data.
"""

import json
import math

from darwinbot.controller import CONTROL_DT, LOAD_FULL_SCALE, RAMP_TIME

FORMAT = 3
CONTROL_HZ = round(1 / CONTROL_DT)   # same as the simulation
ZERO = 2048

_STEER_KEYS = ("steer_gain", "steer_shift")
_SERVO_KEYS = {
    "position": ("id", "name", "mode", "amp1", "phase1", "offset", "amp2", "phase2",
                 "sigma", "sign", "zero", "min", "max") + _STEER_KEYS,
    "wheel": ("id", "name", "mode", "speed", "sign") + _STEER_KEYS,
}
_TOP_KEYS = ("format", "robot", "fitness", "control_hz", "ramp_time", "frequency", "servos")


def make_control_file(robot_name, fitness, layout, params, joint_ranges, signs=None,
                      control_hz=CONTROL_HZ, ramp_time=RAMP_TIME,
                      load_full_scale=LOAD_FULL_SCALE) -> dict:
    """
    Build the control dict. layout items are (slot, kind, axis) in servo order
    (morphology.Actuator), params is controller.Params and signs the mounting
    sign of every servo (default 1). load_full_scale is the fraction of the
    servo maximum torque that the simulation treats as N = 1.
    """
    servos = []
    for i, (slot, kind, axis) in enumerate(layout):
        name = f"s{slot}_{axis}"
        sign = 1 if signs is None else int(signs[i])
        if kind == "wheel":
            servos.append({"id": i + 1, "name": name, "mode": "wheel",
                           "speed": float(params.offset_base[i]), "sign": sign,
                           "steer_gain": float(params.steer_gain[i]),
                           "steer_shift": float(params.steer_shift[i])})
        else:
            lim = float(joint_ranges[axis])
            servos.append({"id": i + 1, "name": name, "mode": "position",
                           "amp1": float(params.amp1_base[i]),
                           "phase1": float(params.phase1[i]),
                           "offset": float(params.offset_base[i]),
                           "amp2": float(params.amp2_base[i]),
                           "phase2": float(params.phase2[i]),
                           "sigma": float(params.sigma[i]),
                           "steer_gain": float(params.steer_gain[i]),
                           "steer_shift": float(params.steer_shift[i]),
                           "sign": sign, "zero": ZERO, "min": -lim, "max": lim})
    return {"format": FORMAT, "robot": robot_name, "fitness": float(fitness),
            "control_hz": control_hz, "ramp_time": ramp_time,
            "load_full_scale": float(load_full_scale),
            "frequency": float(params.omega) / (2 * math.pi), "servos": servos}


def write_control_file(d: dict, path) -> None:
    """Validate d, then write it as indented JSON."""
    _validate(d)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


def read_control_file(path) -> dict:
    """Load and validate a control file (format 2 or 3); ValueError if malformed."""
    with open(path) as f:
        d = json.load(f)
    _validate(d)
    d.setdefault("load_full_scale", LOAD_FULL_SCALE)    # optional field
    for s in d["servos"]:                               # format 2 has no steering
        for k in _STEER_KEYS:
            s.setdefault(k, 0.0)
    return d


def _validate(d) -> None:
    if not isinstance(d, dict):
        raise ValueError("control file must be a JSON object")
    version = d.get("format")
    if version not in (2, FORMAT):
        raise ValueError(f"unsupported control file format {version!r}, "
                         f"only formats 2 and {FORMAT} are supported")
    for k in _TOP_KEYS:
        if k not in d:
            raise ValueError(f"control file missing field '{k}'")
    scale = d.get("load_full_scale", LOAD_FULL_SCALE)
    if not isinstance(scale, (int, float)) or not 0 < scale <= 1:
        raise ValueError(f"load_full_scale must be in (0, 1], got {scale!r}")
    ids = set()
    for s in d["servos"]:
        mode = s.get("mode")
        if mode not in _SERVO_KEYS:
            raise ValueError(f"servo {s.get('id')!r}: unknown mode {mode!r}")
        keys = _SERVO_KEYS[mode]
        if version < 3:                                 # older files lack steering
            keys = tuple(k for k in keys if k not in _STEER_KEYS)
        for k in keys:
            if k not in s:
                raise ValueError(f"servo {s.get('id')!r} missing field '{k}'")
        if s["id"] in ids or not 1 <= s["id"] <= 253:
            raise ValueError(f"servo id {s['id']!r} duplicated or outside 1..253")
        ids.add(s["id"])

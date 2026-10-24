"""
control_file.py
---------------
Writes and reads the JSON control file of docs/formato-controle.md (format 2),
the hand-off from the evolved Tegotae CPG to the ESP32 firmware.

No dependency on morphology or builder: it takes the actuator layout, the
controller parameters and the joint limits as plain data.
"""

import json
import math

from darwinbot.controller import CONTROL_DT, RAMP_TIME

FORMAT = 2
CONTROL_HZ = round(1 / CONTROL_DT)   # same as the simulation
ZERO = 2048

_SERVO_KEYS = {
    "position": ("id", "name", "mode", "amp1", "phase1", "offset", "amp2", "phase2",
                 "sigma", "sign", "zero", "min", "max"),
    "wheel": ("id", "name", "mode", "speed", "sign"),
}
_TOP_KEYS = ("format", "robot", "fitness", "control_hz", "ramp_time", "frequency", "servos")


def make_control_file(robot_name, fitness, layout, params, joint_ranges, signs=None,
                      control_hz=CONTROL_HZ, ramp_time=RAMP_TIME) -> dict:
    """
    Build the control dict. layout items are (slot, kind, axis) in servo order
    (morphology.Actuator), params is controller.Params and signs the mounting
    sign of every servo (default 1).
    """
    servos = []
    for i, (slot, kind, axis) in enumerate(layout):
        name = f"s{slot}_{axis}"
        sign = 1 if signs is None else int(signs[i])
        if kind == "wheel":
            servos.append({"id": i + 1, "name": name, "mode": "wheel",
                           "speed": float(params.offset[i]), "sign": sign})
        else:
            lim = float(joint_ranges[axis])
            servos.append({"id": i + 1, "name": name, "mode": "position",
                           "amp1": float(params.amp1[i]), "phase1": float(params.phase1[i]),
                           "offset": float(params.offset[i]),
                           "amp2": float(params.amp2[i]), "phase2": float(params.phase2[i]),
                           "sigma": float(params.sigma[i]),
                           "sign": sign, "zero": ZERO, "min": -lim, "max": lim})
    return {"format": FORMAT, "robot": robot_name, "fitness": float(fitness),
            "control_hz": control_hz, "ramp_time": ramp_time,
            "frequency": float(params.omega) / (2 * math.pi), "servos": servos}


def write_control_file(d: dict, path) -> None:
    """Validate d, then write it as indented JSON."""
    _validate(d)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


def read_control_file(path) -> dict:
    """Load and validate a control file (format 2 only); ValueError if malformed."""
    with open(path) as f:
        d = json.load(f)
    _validate(d)
    return d


def _validate(d) -> None:
    if not isinstance(d, dict):
        raise ValueError("control file must be a JSON object")
    if d.get("format") != FORMAT:
        raise ValueError(f"unsupported control file format {d.get('format')!r}, "
                         f"only format {FORMAT} is supported")
    for k in _TOP_KEYS:
        if k not in d:
            raise ValueError(f"control file missing field '{k}'")
    ids = set()
    for s in d["servos"]:
        mode = s.get("mode")
        if mode not in _SERVO_KEYS:
            raise ValueError(f"servo {s.get('id')!r}: unknown mode {mode!r}")
        for k in _SERVO_KEYS[mode]:
            if k not in s:
                raise ValueError(f"servo {s.get('id')!r} missing field '{k}'")
        if s["id"] in ids or not 1 <= s["id"] <= 253:
            raise ValueError(f"servo id {s['id']!r} duplicated or outside 1..253")
        ids.add(s["id"])

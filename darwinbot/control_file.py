"""
control_file.py
---------------
Writes and reads the JSON control file of docs/formato-controle.md (format 1),
the hand-off from the evolved CPG to the ESP32 firmware.

No dependency on morphology or builder: it takes the actuator layout, the CPG
arrays and the joint limits as plain data.
"""

import json
import math

from darwinbot.controller import CONTROL_DT, RAMP_TIME

FORMAT = 1
CONTROL_HZ = round(1 / CONTROL_DT)   # same as the simulation
ZERO = 2048

_SERVO_KEYS = {
    "position": ("id", "name", "mode", "amp", "offset", "phase", "sign", "zero", "min", "max"),
    "wheel": ("id", "name", "mode", "speed", "sign"),
}
_TOP_KEYS = ("format", "robot", "fitness", "control_hz", "ramp_time", "frequency", "servos")


def make_control_file(robot_name, fitness, layout, cpg, joint_ranges,
                      wheel_sign_right=-1, control_hz=CONTROL_HZ, ramp_time=RAMP_TIME) -> dict:
    """Build the control dict; layout items are (pair, side, kind), side 0 = left."""
    servos = []
    for i, (pair, side, kind) in enumerate(layout):
        name = f"p{pair}{'LR'[side]}_{kind}"
        if kind == "wheel":
            servos.append({"id": i + 1, "name": name, "mode": "wheel",
                           "speed": float(cpg.offset[i]),
                           "sign": 1 if side == 0 else int(wheel_sign_right)})
        else:
            lim = float(joint_ranges[kind])
            servos.append({"id": i + 1, "name": name, "mode": "position",
                           "amp": float(cpg.amp[i]), "offset": float(cpg.offset[i]),
                           "phase": float(cpg.phase[i]),
                           "sign": 1 if side == 0 else -1, "zero": ZERO,
                           "min": -lim, "max": lim})
    return {"format": FORMAT, "robot": robot_name, "fitness": float(fitness),
            "control_hz": control_hz, "ramp_time": ramp_time,
            "frequency": float(cpg.omega) / (2 * math.pi), "servos": servos}


def write_control_file(d: dict, path) -> None:
    """Validate d, then write it as indented JSON."""
    _validate(d)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


def read_control_file(path) -> dict:
    """Load and validate a control file; ValueError if malformed."""
    with open(path) as f:
        d = json.load(f)
    _validate(d)
    return d


def _validate(d) -> None:
    if not isinstance(d, dict):
        raise ValueError("control file must be a JSON object")
    if d.get("format") != FORMAT:
        raise ValueError(f"unsupported control file format {d.get('format')!r}, expected {FORMAT}")
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

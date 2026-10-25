"""
test_firmware_parity.py
-----------------------
Compile the portable C++ controller core and check, tick by tick, that it
reproduces the targets of darwinbot/controller.py (format 2 firmware rules).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import morphology as mo
from darwinbot.builder import JOINT_RANGE

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "firmware" / "core" / "darwin_control.cpp"
MAIN = ROOT / "firmware" / "test" / "parity_main.cpp"
COMPILERS = ("g++", "clang++", "c++")

LAYOUT = [mo.Actuator(1, "joint", "pitch"),
          mo.Actuator(2, "joint", "yaw"),
          mo.Actuator(3, "joint", "roll"),
          mo.Actuator(4, "wheel", "wheel"),
          mo.Actuator(5, "wheel", "wheel")]
SIGNS = (-1.0, 1.0, -1.0, -1.0, 1.0)      # one reversed wheel, one not
LIMITS = JOINT_RANGE      # same joint limits as the simulation
TICKS = 200


def _fmt(v):
    return f"{float(v):.9g}"


def _write_case(path, ctrl, loads):
    """Write a case file; expected targets come from controller.step()."""
    c = co.Controller(ctrl, LAYOUT)
    p = c.params
    n = len(LAYOUT)
    limits = [LIMITS.get(a.axis, 0.0) for a in LAYOUT]

    lines = [" ".join(_fmt(v) for v in
                      (p.omega / (2 * np.pi), co.RAMP_TIME,
                       round(1 / co.CONTROL_DT), n, co.LOAD_FULL_SCALE))]
    for i, act in enumerate(LAYOUT):
        sign, lim = SIGNS[i], limits[i]
        if act.kind == "wheel":
            fields = (0, 0, 0, 0, 0, 0, sign, 0, 0, 0, p.offset[i])
            lines.append("wheel " + " ".join(_fmt(v) for v in fields))
        else:
            fields = (p.amp1[i], p.phase1[i], p.offset[i], p.amp2[i],
                      p.phase2[i], p.sigma[i], sign, 2048, -lim, lim, 0)
            lines.append("position " + " ".join(_fmt(v) for v in fields))

    for k, load in enumerate(loads):
        # One firmware tick: targets now, then five 4 ms phase substeps with
        # the load held, exactly as docs/formato-controle.md specifies.
        out = c.targets(k * co.CONTROL_DT)
        force = np.asarray(load) * co.SERVO_TORQUE
        for _ in range(5):
            c.advance(co.CONTROL_DT / 5, force)
        expected = [out[i] if act.kind == "wheel"
                    else np.clip(out[i], -limits[i], limits[i])
                    for i, act in enumerate(LAYOUT)]
        speeds = [out[i] * SIGNS[i] if act.kind == "wheel" else 0.0
                  for i, act in enumerate(LAYOUT)]
        # bus load / servo maximum torque: the simulation's full scale is
        # LOAD_FULL_SCALE of it, which the core divides out again
        row = ([load[i] * SIGNS[i] * co.LOAD_FULL_SCALE for i in range(n)]
               + expected + speeds)
        lines.append(" ".join(_fmt(v) for v in row))

    path.write_text("\n".join(lines) + "\n")


def _run_case(tmp_path, ctrl, loads):
    compiler = next((c for c in COMPILERS if shutil.which(c)), None)
    if compiler is None:
        pytest.skip("no C++ compiler found")
    case = tmp_path / "case.txt"
    _write_case(case, ctrl, loads)
    exe = tmp_path / "parity"
    build = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-o", str(exe),
         str(MAIN), str(CORE)],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe), str(case)], capture_output=True, text=True,
                         timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "max abs error" in run.stdout


def _genes(seed):
    rng = np.random.default_rng(seed)
    ctrl = co.random_controller(rng)
    for slot in (1, 2, 3):
        b = co.node_base(slot)
        ctrl[b + co.AMP1] = 0.7
        ctrl[b + co.AMP2] = 0.9          # second harmonic present
        ctrl[b + co.SIGMA] = 0.85        # Tegotae on
    ctrl[co.node_base(4) + co.OFFSET] = 0.7
    ctrl[co.node_base(5) + co.OFFSET] = 0.2      # a slower second wheel
    return rng, ctrl


def test_parity_random_genes_and_loads(tmp_path):
    rng, ctrl = _genes(3)
    p = co.Controller(ctrl, LAYOUT).params
    assert np.all(p.sigma[[0, 1, 2]] != 0)
    assert np.all(p.amp2[[0, 1, 2]] > 0)
    loads = rng.uniform(-1.0, 1.0, size=(TICKS, len(LAYOUT)))
    _run_case(tmp_path, ctrl, loads)


def test_parity_open_loop_when_sigma_is_zero(tmp_path):
    rng, ctrl = _genes(4)
    ctrl[[co.node_base(s) + co.SIGMA for s in (1, 2, 3)]] = 0.5   # sigma = 0
    assert np.all(co.Controller(ctrl, LAYOUT).params.sigma == 0.0)
    loads = rng.uniform(-1.0, 1.0, size=(TICKS, len(LAYOUT)))
    _run_case(tmp_path, ctrl, loads)


def test_parity_with_speed_clip_active(tmp_path):
    rng, ctrl = _genes(5)
    ctrl[0] = 1.0                        # 3 Hz
    for slot in (1, 2, 3):
        b = co.node_base(slot)
        ctrl[b + co.AMP1] = 1.0
        ctrl[b + co.AMP2] = 1.0
    p = co.Controller(ctrl, LAYOUT).params
    peak = (p.amp1 + 2 * p.amp2) * p.omega
    assert np.allclose(peak[[0, 1, 2]], co.SERVO_SPEED)
    loads = rng.uniform(-1.0, 1.0, size=(TICKS, len(LAYOUT)))
    _run_case(tmp_path, ctrl, loads)


def test_wheel_speed_signs_reach_the_output(tmp_path):
    rng, ctrl = _genes(6)
    p = co.Controller(ctrl, LAYOUT).params
    assert p.offset[3] > 0 and SIGNS[3] < 0 and SIGNS[4] > 0
    loads = rng.uniform(-1.0, 1.0, size=(TICKS, len(LAYOUT)))
    _run_case(tmp_path, ctrl, loads)

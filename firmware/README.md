# Firmware

Portable control math for the real DarwinBot, in plain C++17 (no Arduino, no
external libraries, float math), so it can be tested without the ESP32.

`core/darwin_control.{h,cpp}` implements the format 2 controller of
`docs/formato-controle.md`: one phase oscillator per position servo with
Tegotae load feedback (`dtheta = omega - sigma * N * cos(theta)`, five
substeps per control tick), the start ramp, the speed clip
`(amp1 + 2 * amp2) * omega <= 5.45 rad/s`, joint clamping and the servo step
conversion (`zero + sign * target * 4096 / 2pi`, rounded). Wheels get only a
ramped, signed speed and ignore phase and load.

The core takes loads as bus load over the servo maximum torque and divides
them by `load_full_scale` (control.json, default 0.7: the simulation treats
70% of the stall torque as N = 1). The phase is wrapped to [0, 2pi) every tick
so float precision never stalls it.

## Parity test

`tests/test_firmware_parity.py` builds a case with `numpy` from
`darwinbot/controller.py` (random genes, open loop with `sigma = 0`, an
active speed clip, and signed wheel speeds), writes it to a temporary file, compiles the core plus
`test/parity_main.cpp` with the first of `g++`, `clang++`, `c++` found, and
runs it. The test skips only when no compiler is installed.

```bash
pytest tests/test_firmware_parity.py -q
```

### Running a case by hand

`test/parity_main.cpp` reads a plain text case file. Its header comment
documents the format: a line with `frequency ramp_time control_hz n_servos`,
an optional `load_full_scale` on the header line, one line per servo, then one
line per tick with the loads, the expected targets and the expected signed
wheel speeds. Expected targets are radians for position servos and
rad/s for wheels.

```bash
g++ -std=c++17 -O2 firmware/test/parity_main.cpp \
    firmware/core/darwin_control.cpp -o /tmp/parity
/tmp/parity case.txt
```

It prints the number of ticks and the maximum absolute error and exits 1 when
that error exceeds `1e-3 rad`.

## ESP32 layer

Not written yet. When the hardware arrives, a thin Arduino layer will read the
control JSON, poll each STS3215 load register (`0x3C`/`0x3D`), apply the
calibration (`zero`, `sign`) and write targets on the bus. All control math
stays in this portable core.

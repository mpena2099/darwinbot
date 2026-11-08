"""
power.py
--------
Electrical estimate of a robot: the current every servo draws in the nominal
simulation, its peaks, the runtime on the 3S pack and the C rating the pack
needs. A report only: nothing here reaches the score.

The current model is linear between the idle and the stall current of the
12 V STS3215 (docs/hardware-dados.md), driven by the actuator force MuJoCo
reports for that servo (wheels are servos too).
"""

from dataclasses import dataclass

import numpy as np

from darwinbot.controller import SERVO_TORQUE

# Feetech STS3215, 12 V (C018), docs/hardware-dados.md: 200 mA idle, 2.7 A
# stall. The stall torque is the limit the servo model already uses
# (controller.SERVO_TORQUE: the 2.94 N*m of the table with its 30% margin).
I_IDLE = 0.2                  # A, "corrente em repouso"
I_STALL = 2.7                 # A, "corrente de travamento"
TAU_STALL = SERVO_TORQUE      # N*m, force where the current saturates
# The doc has no ESP32 figure: 0.15 A for the board and its radio.
ELECTRONICS_A = 0.15          # A
# The doc's battery is a 3S LiPo (11.1 V) of 1000 to 1500 mAh and lists no C
# rating: take the middle of the range and a common pack rating.
BATTERY_AH = 1.2              # Ah, nominal capacity of the documented range
BATTERY_C = 20.0              # discharge rating of a pack that size
BATTERY_V = 11.1              # V nominal


def servo_current(tau):
    """Current of one servo at torque `tau` (N*m), clipped at stall: scalar or
    array, in amperes."""
    load = np.minimum(1.0, np.abs(tau) / TAU_STALL)
    return I_IDLE + (I_STALL - I_IDLE) * load


@dataclass
class PowerEstimate:
    n_servos:        int
    mean_a:          float     # mean total current over the run
    peak_a:          float     # highest total current of any physics step
    peak_1s_a:       float     # highest 1 s rolling mean of the total
    per_servo_peak_a: list     # highest current of each servo, layout order
    runtime_min:     float     # BATTERY_AH / mean_a, in minutes
    c_needed:        float     # peak_1s_a / BATTERY_AH
    capacity_ah:     float = BATTERY_AH
    c_rating:        float = BATTERY_C
    volts:           float = BATTERY_V


def estimate(episode, ctrl, seconds: float, command: float = 0.0) -> PowerEstimate:
    """Run the nominal episode (as experiments/replay.py record does) and read
    the actuator force of every servo at every physics step."""
    cpg = episode.reset(ctrl, command=command)
    dt = episode.model.opt.timestep
    n = max(1, int(round(seconds / dt)))
    total = np.zeros(n)
    per_peak = np.zeros(episode.model.nu)
    for k in range(n):
        episode.step(cpg)
        current = np.nan_to_num(servo_current(episode.data.actuator_force))
        total[k] = current.sum() + ELECTRONICS_A
        np.maximum(per_peak, current, out=per_peak)
    mean = float(total.mean())
    # 1 s rolling mean, O(n) with the cumulative sum (shape keeps the tail:
    # a short run gives the mean of what ran).
    window = max(1, int(round(1.0 / dt)))
    cum = np.concatenate(([0.0], np.cumsum(total)))
    sums = cum[window:] - cum[:-window]
    peak_1s = float(sums.max() / window) if len(sums) else mean
    return PowerEstimate(
        n_servos=episode.model.nu,
        mean_a=mean,
        peak_a=float(total.max()),
        peak_1s_a=peak_1s,
        per_servo_peak_a=[float(v) for v in per_peak],
        runtime_min=BATTERY_AH / mean * 60.0 if mean > 0 else float("inf"),
        c_needed=peak_1s / BATTERY_AH,
    )


def report_lines(result: PowerEstimate) -> list:
    """Short report of an estimate; the last line says where it comes from."""
    pack = result.capacity_ah * result.c_rating
    lines = [
        f"Power ({result.n_servos} servos): mean {result.mean_a:.2f} A, "
        f"peak {result.peak_a:.2f} A, 1 s peak {result.peak_1s_a:.2f} A",
        f"runtime {result.runtime_min:.0f} min on a "
        f"{result.capacity_ah * 1000:.0f} mAh 3S LiPo ({result.volts:.1f} V, "
        f"{result.c_rating:.0f} C)",
        f"needs {result.c_needed:.1f} C; the pack gives {result.c_rating:.0f} C "
        f"({pack:.1f} A)",
    ]
    if result.peak_1s_a > pack:
        lines.append(f"WARNING: the 1 s peak of {result.peak_1s_a:.1f} A is above "
                     f"what the pack can deliver ({pack:.1f} A): pick a pack "
                     f"with a higher C rating or a larger capacity")
    lines.append("estimate from simulated torque, not measured")
    return lines

"""Optional score rules that short experiment runs can switch on, one at a
time, to measure what each does to the archive before it enters a run. None is
active by default, so the score is the one documented in `fitness.py`.

The set is module-level state: `experiments/train.py --rules NAME` stores it in
the archive, and the training pool workers call `set_active` on start. The
metrics the rules read (`tilt`, `slip`, `servo_ground`) are always measured, in
`sim.py` and `sim_gpu.py`; the rules only decide whether they enter the score.

  worse_turn    steer task: the worse of the two turns counts, not their mean
  turn_forward  steer task: a turn earns credit only if the torso moved forward
  tilt          subtract TILT_W * mean torso tilt (rad) from every episode
  slip          subtract SLIP_W * mean speed of feet sliding while in contact
  servo_ground  subtract SERVO_GROUND_W * share of control ticks a servo
                housing touches the ground
  body_ground   subtract BODY_GROUND_W * share of control ticks the torso
                itself touches the ground
  energy_abs    subtract ENERGY_ABS_W * energy / (M_REF g): the servo work
                itself, not divided by the robot's own mass, so dead weight costs
  basal         subtract BASAL_PER_SERVO * n_servos + BASAL_PER_KG * mass (kg)
                once per evaluation: the idle current of every servo and the
                weight every gram adds, as a living body's basal metabolism
  fall_zero     an episode that ends in a fall earns no progress credit (only
                FALL_PENALTY is left), in every task
"""
import os
from typing import Iterable

NAMES = ("worse_turn", "turn_forward", "tilt", "slip", "servo_ground",
         "body_ground", "energy_abs", "basal", "fall_zero")

# Score units (metres) per servo and per kg of rule `basal`, tuned by the gate.
# DARWINBOT_BASAL_SCALE multiplies both, for the strength gate (1 = these values).
BASAL_SCALE = float(os.environ.get("DARWINBOT_BASAL_SCALE", "1"))
BASAL_PER_SERVO = 0.01 * BASAL_SCALE
BASAL_PER_KG = 0.05 * BASAL_SCALE

_active: frozenset = frozenset()


def normalise(names: Iterable[str]) -> tuple:
    names = tuple(names or ())
    bad = [n for n in names if n not in NAMES]
    if bad:
        raise ValueError(f"unknown rule {bad}, expected some of {list(NAMES)}")
    return tuple(n for n in NAMES if n in names)


def set_active(names: Iterable[str]) -> None:
    global _active
    _active = frozenset(normalise(names))


def active(name: str) -> bool:
    return name in _active

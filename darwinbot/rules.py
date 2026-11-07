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
"""
from typing import Iterable

NAMES = ("worse_turn", "turn_forward", "tilt", "slip", "servo_ground")

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

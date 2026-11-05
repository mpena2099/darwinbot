"""Realism rules that short experiment runs can switch off, one at a time, to
measure which rule costs the robots the most movement. Never used by default.

The set is module-level state: `experiments/train.py --ablate NAME` stores it in
the archive, and the training pool workers call `set_active` on start. CPU
simulation only (`sim.py`); `sim_gpu.py` does not read it.

  ranges    joints keep the default JOINT_RANGE (builder.joint_ranges not applied)
  friction  hard plastic grips like TPU (PLASTIC_FACTOR = 1)
  assembly  printed-part overlaps are allowed at the spawn (only servo overlaps count)
  forward   the forward task replaces steer (applied by train.py, not read here)
"""
from typing import Iterable

NAMES = ("ranges", "friction", "assembly", "forward")

_active: frozenset = frozenset()


def normalise(names: Iterable[str]) -> tuple:
    names = tuple(names or ())
    bad = [n for n in names if n not in NAMES]
    if bad:
        raise ValueError(f"unknown ablation {bad}, expected some of {list(NAMES)}")
    return tuple(n for n in NAMES if n in names)


def set_active(names: Iterable[str]) -> None:
    global _active
    _active = frozenset(normalise(names))


def active(name: str) -> bool:
    return name in _active

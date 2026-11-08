"""Limits on the bodies evolution may build, so a run can be pointed at the
robot the builder wants: no wheels, at most eight servos, four to six legs, a
small printer.

The set is module-level state: `experiments/train.py --body "no-wheels,legs=4..6"`
stores the string in the archive, and the training pool workers, replay,
validate, export, polish and prune call `set_active` from the archive before
they decode or mutate a body. Nothing in the score reads it: a body the options
forbid is never made, and `morphology.violations` rejects one that still breaks
a limit, so it is never simulated.

  no-wheels         no wheel modules
  no-passive        no spring, tail, skid, spike or caster
  no-segments       no spine segments
  max-servos=N      at most N servos (joints, wheels and segments)
  legs=A..B         between A and B legs (legs=N means exactly N)
  max-size=MM       largest bounding-box side of the body at spawn, in mm
  bed=MM            print bed side used by the printability checks, in mm
"""

from dataclasses import dataclass, replace
from typing import Optional

from darwinbot import morphology as morpho

PASSIVE_KINDS = ("spring", "tail", "skid", "spike", "caster")
BED_MM = float(round(morpho.BED * 1000.0, 6))


@dataclass(frozen=True)
class BodyOptions:
    allow_wheels: bool = True
    allow_passive: bool = True
    allow_segments: bool = True
    max_servos: int = morpho.MAX_SERVOS
    min_legs: int = 0
    max_legs: Optional[int] = None
    max_size_mm: Optional[float] = None
    bed_mm: float = BED_MM

    def __post_init__(self):
        if self.max_servos < 1:
            raise ValueError("max-servos must be at least 1 (a body needs a servo)")
        if self.min_legs < 0 or (self.max_legs is not None and self.max_legs < 0):
            raise ValueError("legs must not be negative")
        if self.max_legs is not None and self.max_legs < self.min_legs:
            raise ValueError("legs range ends before it starts")
        if self.min_legs and self.max_legs is None:
            raise ValueError("a minimum number of legs needs an upper bound: legs=A..B")
        if self.max_size_mm is not None and self.max_size_mm <= 0:
            raise ValueError("max-size must be positive")
        if self.bed_mm <= 0:
            raise ValueError("bed must be positive")

    def allows(self, kind: str) -> bool:
        """True when a module of this kind may be built."""
        if kind == "wheel":
            return self.allow_wheels
        if kind in PASSIVE_KINDS:
            return self.allow_passive
        if kind == "segment":
            return self.allow_segments
        return True

    def is_default(self) -> bool:
        return self == BodyOptions()

    def to_string(self) -> str:
        """Canonical comma list; `parse` turns it back into the same options."""
        out = []
        if not self.allow_wheels:
            out.append("no-wheels")
        if not self.allow_passive:
            out.append("no-passive")
        if not self.allow_segments:
            out.append("no-segments")
        if self.max_servos != morpho.MAX_SERVOS:
            out.append(f"max-servos={self.max_servos}")
        if self.max_legs is not None:
            out.append(f"legs={self.min_legs}"
                       if self.max_legs == self.min_legs
                       else f"legs={self.min_legs}..{self.max_legs}")
        if self.max_size_mm is not None:
            out.append(f"max-size={_num(self.max_size_mm)}")
        if self.bed_mm != BED_MM:
            out.append(f"bed={_num(self.bed_mm)}")
        return ",".join(out)


def _num(value: float) -> str:
    return f"{value:g}"


def _int(name: str, text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"bad value '{text}' for {name}, expected a whole number")


def _float(name: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"bad value '{text}' for {name}, expected a number")
    if value <= 0:
        raise ValueError(f"bad value '{text}' for {name}, expected a positive number")
    return value


FLAGS = {"no-wheels": "allow_wheels", "no-passive": "allow_passive",
         "no-segments": "allow_segments"}


def parse(text: str) -> BodyOptions:
    """`experiments/train.py --body` string -> BodyOptions. An empty string is
    the defaults, exactly the bodies evolution made before this option."""
    opts = BodyOptions()
    for token in (text or "").split(","):
        token = token.strip()
        if not token:
            continue
        if token in FLAGS:
            opts = replace(opts, **{FLAGS[token]: False})
            continue
        if "=" not in token:
            raise ValueError(_unknown(token))
        name, value = (part.strip() for part in token.split("=", 1))
        name = name.lower()
        if name == "max-servos":
            opts = replace(opts, max_servos=_int(name, value))
        elif name == "legs":
            low, _, high = value.partition("..")
            low = _int(name, low)
            high = low if not high else _int(name, high)
            if low < 0 or high < low:
                raise ValueError(f"bad legs range '{value}', expected legs=A..B with 0 <= A <= B")
            opts = replace(opts, min_legs=low, max_legs=high)
        elif name == "max-size":
            opts = replace(opts, max_size_mm=_float(name, value))
        elif name == "bed":
            opts = replace(opts, bed_mm=_float(name, value))
        else:
            raise ValueError(_unknown(token))
    return opts


def _unknown(token: str) -> str:
    return (f"unknown body option '{token}', expected some of no-wheels, "
            "no-passive, no-segments, max-servos=N, legs=A..B, max-size=MM, bed=MM")


_active = BodyOptions()


def set_active(opts) -> None:
    """Set the options the body makers read; None restores the defaults."""
    global _active
    _active = parse(opts) if isinstance(opts, str) else (opts or BodyOptions())


def active() -> BodyOptions:
    return _active

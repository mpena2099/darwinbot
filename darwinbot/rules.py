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
                weight every gram adds, as a living body's basal metabolism.
                --basal-mode relative instead keeps (1 - f) of a positive
                score, f = servo share * servos + part share * passive parts,
                capped at BASAL_REL_CAP
  fall_zero     an episode that ends in a fall earns no progress credit (only
                FALL_PENALTY is left), in every task
"""
import os
from typing import Iterable

NAMES = ("worse_turn", "turn_forward", "tilt", "slip", "servo_ground",
         "body_ground", "energy_abs", "basal", "fall_zero")

# Score units (metres) per servo and per kg of rule `basal`, tuned by the gate.
# DARWINBOT_BASAL_SCALE multiplies both, for the strength gate (1 = these values);
# train.py --basal-scale stores the same setting in the archive.
DEFAULT_BASAL_SCALE = 1.0
BASAL_MODES = ("absolute", "relative")
DEFAULT_BASAL_MODE = "absolute"
DEFAULT_BASAL_SERVO_SHARE = 0.04   # score fraction per servo, relative mode
DEFAULT_BASAL_PART_SHARE = 0.02    # score fraction per passive part, relative mode
BASAL_REL_CAP = 0.8               # most of the score the relative rule removes


def _scale(value=None) -> float:
    """The environment wins while set (gates), then an explicit value."""
    if "DARWINBOT_BASAL_SCALE" in os.environ:
        return float(os.environ["DARWINBOT_BASAL_SCALE"])
    return DEFAULT_BASAL_SCALE if value is None else float(value)


def _mode(value=None) -> str:
    if "DARWINBOT_BASAL_MODE" in os.environ:
        return os.environ["DARWINBOT_BASAL_MODE"]
    return DEFAULT_BASAL_MODE if value is None else value


def _share(env: str, default: float, value=None) -> float:
    if env in os.environ:
        return float(os.environ[env])
    return default if value is None else float(value)


BASAL_SCALE = _scale()
BASAL_PER_SERVO = 0.01 * BASAL_SCALE
BASAL_PER_KG = 0.05 * BASAL_SCALE
BASAL_MODE = _mode()
BASAL_SERVO_SHARE = _share("DARWINBOT_BASAL_SERVO_SHARE", DEFAULT_BASAL_SERVO_SHARE)
BASAL_PART_SHARE = _share("DARWINBOT_BASAL_PART_SHARE", DEFAULT_BASAL_PART_SHARE)


def score_option(value=None, mode=None, servo_share=None, part_share=None) -> dict:
    """Non-default basal settings for Archive.score (env wins while set)."""
    out = {}
    scale = _scale(value)
    if scale != DEFAULT_BASAL_SCALE:
        out["basal_scale"] = scale
    basal_mode = _mode(mode)
    if basal_mode != DEFAULT_BASAL_MODE:
        out["basal_mode"] = basal_mode
    share = _share("DARWINBOT_BASAL_SERVO_SHARE", DEFAULT_BASAL_SERVO_SHARE, servo_share)
    if share != DEFAULT_BASAL_SERVO_SHARE:
        out["basal_servo_share"] = share
    share = _share("DARWINBOT_BASAL_PART_SHARE", DEFAULT_BASAL_PART_SHARE, part_share)
    if share != DEFAULT_BASAL_PART_SHARE:
        out["basal_part_share"] = share
    return out


def apply_score(score: dict) -> None:
    """Use an archive's stored settings; the environment overrides them when set."""
    global BASAL_SCALE, BASAL_PER_SERVO, BASAL_PER_KG, BASAL_MODE
    global BASAL_SERVO_SHARE, BASAL_PART_SHARE
    BASAL_SCALE = _scale(score.get("basal_scale"))
    BASAL_PER_SERVO = 0.01 * BASAL_SCALE
    BASAL_PER_KG = 0.05 * BASAL_SCALE
    BASAL_MODE = _mode(score.get("basal_mode"))
    BASAL_SERVO_SHARE = _share("DARWINBOT_BASAL_SERVO_SHARE",
                               DEFAULT_BASAL_SERVO_SHARE, score.get("basal_servo_share"))
    BASAL_PART_SHARE = _share("DARWINBOT_BASAL_PART_SHARE",
                              DEFAULT_BASAL_PART_SHARE, score.get("basal_part_share"))


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

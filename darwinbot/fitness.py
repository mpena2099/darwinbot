"""
fitness.py
----------
Episode metrics -> scalar score, and the multi-terrain aggregate.

score = up(progress) - LATERAL_W * |dy| - HEADING_W * |yaw| - ENERGY_W * energy / (m g) - FALL_PENALTY * fell

progress counts only the second half of the episode, scaled by 2:
2 * (x_end - x_mid) along the heading axis (Metrics.progress). A steady gait
keeps its distance, while a one-time drop of the torso early on is mostly
ignored. If the episode ends in a fall before the half-way step, progress is
0. A fall after the mark keeps 2 * (x_fall - x_mid). dy, yaw, energy, the upright factor and the fall
penalty use the whole episode.

up(v) is the progress v times the mean cosine of the torso tilt over the
episode when v is positive: a robot that leans 45 degrees keeps 71% of its
distance, one that leans 10 degrees keeps 98%.

energy / (m g) is in metres: the height the robot could have lifted itself
with the same work. Dividing by weight keeps heavy and light bodies on the
same footing, so the term behaves like a cost of transport.

|yaw| is the heading change over the episode in radians; HEADING_W charges the
same metres per radian that a turning episode earns (TURN_SCALE).

For the "turn" task the goal is rotation in place instead of distance:

score = up(|yaw|) - TURN_DRIFT_W * hypot(dx, dy) - ENERGY_W * energy / (m g) - FALL_PENALTY * fell

|yaw| is in radians and the sign does not matter, so a body may turn
whichever way suits it. Drifting away from the spawn point costs score.

For the "steer" task every evaluation runs four episodes under the (steering,
speed) commands of COMMANDS: straight at full speed, left (-1), right (+1), and
straight at half speed (0.5). The straight one
scores as "forward"; a turning one scores the heading change towards the side
asked for (yaw is positive to the left), TURN_SCALE metres per radian, up to
TURN_CAP, and never more than the distance the torso travelled in that
episode (then up() of that), minus the energy and fall terms:

score = 0.4 * straight + 0.2 * (left + right) + 0.2 * half

A fall ends the episode; the progress rule above says what it keeps.

The half-speed episode asks for half the distance of the full-speed one. With P
the upright-weighted progress of an episode (never below 0):

half = 0.5 * P_full - |P_half - 0.5 * P_full| - energy cost - FALL_PENALTY * fell

It earns the most when the body walks at exactly half pace, nothing when it
ignores the command (P_half = P_full) or stands still, and a faster walker has
more to gain. The speed genes get no credit unless they work.

The turn credit and the turn task's drift use the whole episode (hypot(dx, dy)).

Optional rules (darwinbot/rules.py, off by default, switched on per run with
`train.py --rules`; each is meant to be tried alone in a short run first):

  worse_turn    steer: 0.5 * straight + 0.5 * min(left, right)
  turn_forward  steer: a turn's distance cap is zero when the torso ended
                behind the spawn (dx <= 0), so a curve driven in reverse earns nothing
  tilt          every episode: - TILT_W * Metrics.tilt (mean tilt, rad)
  slip          every episode: - SLIP_W * Metrics.slip (m/s of sliding feet)
  servo_ground  every episode: - SERVO_GROUND_W * Metrics.servo_ground (share of ticks)
  body_ground   every episode: - BODY_GROUND_W * Metrics.body_ground (share of ticks)
  energy_abs    every episode: - ENERGY_ABS_W * energy / (M_REF g), M_REF = 1 kg
  basal         once per evaluation: - BASAL_PER_SERVO * n_servos
                - BASAL_PER_KG * mass (kg), the idle draw and the weight carried
  fall_zero     an episode that fell: its progress credit (up(progress) or
                up(|yaw|), and the half-speed credit) is 0; FALL_PENALTY stays
"""

import math
import os
from typing import Sequence

import numpy as np

from darwinbot import rules
from darwinbot.sim import Metrics

LATERAL_W    = 0.5
ENERGY_W     = 0.05
ENERGY_ABS_W = 0.05   # rule `energy_abs`: m per (energy / (M_REF g))
M_REF        = 1.0    # kg: reference mass of rule `energy_abs`
FALL_PENALTY = 0.5
TURN_DRIFT_W = 0.5
BUILTIN_TASKS = ("forward", "turn", "steer")
TASKS        = BUILTIN_TASKS   # known task names; register_task() keeps it updated
SPEED_ASKED  = 0.5     # speed command of the slow episode of the steer task
COMMANDS     = {"steer": ((0.0, 1.0), (-1.0, 1.0), (1.0, 1.0), (0.0, SPEED_ASKED))}
STRAIGHT     = ((0.0, 1.0),)   # other tasks: one straight episode, full speed
# m per radian of heading turned the way asked (at 0.2 a full 90-degree turn
# earns 0.31 m against 2.5-4 m of straight progress). train.py stores these
# settings in the archive; the env variables win while set (the turn gates).
DEFAULT_TURN_SCALE = 0.2
DEFAULT_TURN_IN_PLACE = False
DEFAULT_TURN_CAP_DEG = 90.0


def _turn_scale(value=None) -> float:
    if "DARWINBOT_TURN_SCALE" in os.environ:
        return float(os.environ["DARWINBOT_TURN_SCALE"])
    return DEFAULT_TURN_SCALE if value is None else float(value)


def _turn_in_place(value=None) -> bool:
    if "DARWINBOT_TURN_IN_PLACE" in os.environ:
        return os.environ["DARWINBOT_TURN_IN_PLACE"] == "1"
    if value is None:
        return DEFAULT_TURN_IN_PLACE
    return value == "1" if isinstance(value, str) else bool(value)


def _turn_cap(value=None) -> float:
    """Radians; `value` and the env variable are in degrees."""
    if "DARWINBOT_TURN_CAP_DEG" in os.environ:
        return math.radians(float(os.environ["DARWINBOT_TURN_CAP_DEG"]))
    return math.radians(DEFAULT_TURN_CAP_DEG if value is None else float(value))


TURN_SCALE   = _turn_scale()
HEADING_W    = 0.2     # m per radian of heading lost on a straight walk
TURN_IN_PLACE = _turn_in_place()
TURN_CAP     = _turn_cap()


def score_options(turn_scale=None, turn_in_place=None, turn_cap_deg=None) -> dict:
    """Non-default turn settings for Archive.score (env wins while set)."""
    out = {}
    scale = _turn_scale(turn_scale)
    if scale != DEFAULT_TURN_SCALE:
        out["turn_scale"] = scale
    in_place = _turn_in_place(turn_in_place)
    if in_place != DEFAULT_TURN_IN_PLACE:
        out["turn_in_place"] = in_place
    if "DARWINBOT_TURN_CAP_DEG" in os.environ:
        deg = float(os.environ["DARWINBOT_TURN_CAP_DEG"])
    else:
        deg = DEFAULT_TURN_CAP_DEG if turn_cap_deg is None else float(turn_cap_deg)
    if deg != DEFAULT_TURN_CAP_DEG:
        out["turn_cap_deg"] = deg
    return out


def apply_score(score: dict) -> None:
    """Use an archive's stored settings; the environment overrides them when set."""
    global TURN_SCALE, TURN_IN_PLACE, TURN_CAP
    TURN_SCALE = _turn_scale(score.get("turn_scale"))
    TURN_IN_PLACE = _turn_in_place(score.get("turn_in_place"))
    TURN_CAP = _turn_cap(score.get("turn_cap_deg"))


TILT_W         = 0.2   # m per radian of mean tilt (rule `tilt`): 10 degrees costs 0.035 m
SLIP_W         = 0.2   # m per m/s of summed foot sliding speed (rule `slip`)
SERVO_GROUND_W = 0.1   # m per share of ticks a servo housing is on the ground (rule `servo_ground`)
BODY_GROUND_W  = 0.2   # m per share of ticks the torso itself is on the ground (rule `body_ground`)
UNSTABLE_SCORE = -5.0
G = 9.81

_TASK_SCORES: dict = {}      # name -> score_fn of a registered plugin task
_TASK_COMMANDS: dict = {}    # name -> commands of a registered plugin task


def task_names() -> tuple:
    """Every known task: the built-ins first, then the registered plugins."""
    return BUILTIN_TASKS + tuple(_TASK_SCORES)


def register_task(name, score_fn=None, commands=STRAIGHT):
    """Register an external task, loaded by a run's --plugin file.

    Use as `register_task("name", fn)` or as a decorator `@register_task("name")`.
    score_fn takes one Metrics, or the list of Metrics (one per command) when
    the task runs more than one command. A built-in name cannot be replaced.
    """
    def add(fn):
        if name in BUILTIN_TASKS:
            raise ValueError(f"cannot overwrite built-in task '{name}'")
        _TASK_SCORES[name] = fn
        _TASK_COMMANDS[name] = tuple(commands)
        global TASKS
        TASKS = task_names()
        return fn
    return add if score_fn is None else add(score_fn)


def up(v: float, m: Metrics) -> float:
    """Progress scaled by how upright the torso stayed; losses stay whole."""
    return v * m.upright if v > 0 else v


def credit(v: float, m: Metrics) -> float:
    """up(v), or 0 for a fallen episode under rule `fall_zero`."""
    if m.fell and rules.active("fall_zero"):
        return 0.0
    return up(v, m)


def rule_cost(m: Metrics) -> float:
    """Penalty of the active optional rules (0 when none is on)."""
    c = 0.0
    if rules.active("tilt"):
        c += TILT_W * m.tilt
    if rules.active("slip"):
        c += SLIP_W * m.slip
    if rules.active("servo_ground"):
        c += SERVO_GROUND_W * m.servo_ground
    if rules.active("body_ground"):
        c += BODY_GROUND_W * m.body_ground
    if rules.active("energy_abs"):
        # the energy term divides by the robot's own mass, so a heavy idle part
        # lowers it; this charges the energy itself, against a fixed mass
        c += ENERGY_ABS_W * m.energy / (M_REF * G)
    return c


def basal_cost(ms: Sequence[Metrics]) -> float:
    """Rule `basal`: what the body costs while it stands still, once per
    evaluation (every episode of a task runs the same body)."""
    if not rules.active("basal") or any(m.unstable for m in ms):
        return 0.0
    m = ms[0]
    return rules.BASAL_PER_SERVO * m.n_servos + rules.BASAL_PER_KG * m.mass


def score(m: Metrics, task: str = "forward") -> float:
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}', expected one of {TASKS}")
    if m.unstable:
        return UNSTABLE_SCORE
    if task not in BUILTIN_TASKS:
        return float(_TASK_SCORES[task](m))
    cost = ENERGY_W * m.energy / (m.mass * G) + rule_cost(m)
    if task == "turn":
        s = credit(abs(m.yaw), m) - TURN_DRIFT_W * math.hypot(m.dx, m.dy) - cost
    else:
        s = credit(m.progress, m) - LATERAL_W * abs(m.dy) - HEADING_W * abs(m.yaw) - cost
    if m.fell:
        s -= FALL_PENALTY
    return float(s)


def commands(task: str):
    """(steering, speed) commands every evaluation runs, in order."""
    return _TASK_COMMANDS.get(task, COMMANDS.get(task, STRAIGHT))


def task_score(ms: Sequence[Metrics], task: str = "forward") -> float:
    """Score of one terrain from the episodes run under commands(task)."""
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}', expected one of {TASKS}")
    if task != "steer":
        if task in BUILTIN_TASKS:
            return float(score(ms[0], task) - basal_cost(ms))
        if any(m.unstable for m in ms):
            return UNSTABLE_SCORE
        fn = _TASK_SCORES[task]
        s = fn(ms) if len(commands(task)) > 1 else fn(ms[0])
        return float(s - basal_cost(ms))
    if any(m.unstable for m in ms):
        return UNSTABLE_SCORE
    turns = []
    for m, (c, _) in zip(ms[1:3], commands(task)[1:3]):
        dist = math.hypot(m.dx, m.dy)
        if rules.active("turn_forward") and m.dx <= 0:
            dist = 0.0
        turn = TURN_SCALE * min(-c * m.yaw, TURN_CAP)
        # default: a turn earns at most the distance walked in it; with TURN_IN_PLACE a
        # pivot earns too, but never more than the straight walk, so a robot that does
        # not walk forward earns nothing for turning (turning is secondary)
        turn = min(turn, max(ms[0].progress, 0.0) if TURN_IN_PLACE else dist)
        s = credit(turn, m) - ENERGY_W * m.energy / (m.mass * G) - rule_cost(m)
        turns.append(s - FALL_PENALTY if m.fell else s)
    turn = min(turns) if rules.active("worse_turn") else np.mean(turns)
    full = max(credit(ms[0].progress, ms[0]), 0.0)
    slow = max(credit(ms[3].progress, ms[3]), 0.0)
    half = (SPEED_ASKED * full - abs(slow - SPEED_ASKED * full)
            - ENERGY_W * ms[3].energy / (ms[3].mass * G) - rule_cost(ms[3]))
    if ms[3].fell:
        half -= FALL_PENALTY
    return float(0.4 * score(ms[0], "forward") + 0.4 * turn + 0.2 * half
                 - basal_cost(ms))


def aggregate(scores: Sequence[float]) -> float:
    """
    Mean over terrains, blended with the worst one. Pure mean lets a robot
    ignore a terrain it fails on; the min term rewards robustness.
    """
    scores = np.asarray(scores, dtype=float)
    return float(0.7 * scores.mean() + 0.3 * scores.min())

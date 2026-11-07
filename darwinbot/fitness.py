"""
fitness.py
----------
Episode metrics -> scalar score, and the multi-terrain aggregate.

score = up(progress) - LATERAL_W * |dy| - HEADING_W * |yaw| - ENERGY_W * energy / (m g) - FALL_PENALTY * fell

progress counts only the second half of the episode, scaled by 2:
2 * (x_end - x_mid) along the heading axis (Metrics.progress). A steady gait
keeps its distance, while a one-time drop of the torso early on is mostly
ignored. If the episode ends before the half-way step (a fall), the distance
covered so far counts. dy, yaw, energy, the upright factor and the fall
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

For the "steer" task every evaluation runs three episodes under the steering
commands of COMMANDS: straight (0), left (-1) and right (+1). The straight one
scores as "forward"; a turning one scores the heading change towards the side
asked for (yaw is positive to the left), TURN_SCALE metres per radian, up to
TURN_CAP, and never more than the distance the torso travelled in that
episode (then up() of that), minus the energy and fall terms:

score = 0.5 * straight + 0.25 * (left + right)

A fall ends the episode but keeps the distance covered so far.

Optional rules (darwinbot/rules.py, off by default, switched on per run with
`train.py --rules`; each is meant to be tried alone in a short run first):

  worse_turn    steer: 0.5 * straight + 0.5 * min(left, right)
  turn_forward  steer: a turn's distance cap is zero when the torso ended
                behind the spawn (dx <= 0), so a curve driven in reverse earns nothing
  tilt          every episode: - TILT_W * Metrics.tilt (mean tilt, rad)
  slip          every episode: - SLIP_W * Metrics.slip (m/s of sliding feet)
  servo_ground  every episode: - SERVO_GROUND_W * Metrics.servo_ground (share of ticks)
"""

import math
from typing import Sequence

import numpy as np

from darwinbot import rules
from darwinbot.sim import Metrics

LATERAL_W    = 0.5
ENERGY_W     = 0.05
FALL_PENALTY = 0.5
TURN_DRIFT_W = 0.5
TASKS        = ("forward", "turn", "steer")
COMMANDS     = {"steer": (0.0, -1.0, 1.0)}   # other tasks: one straight episode
TURN_SCALE   = 0.2     # m per radian of heading turned the way asked
HEADING_W    = TURN_SCALE   # m per radian of heading lost on a straight walk
TURN_CAP     = math.pi / 2   # rad: no credit for turning more than 90 degrees
TILT_W         = 0.2   # m per radian of mean tilt (rule `tilt`): 10 degrees costs 0.035 m
SLIP_W         = 0.2   # m per m/s of summed foot sliding speed (rule `slip`)
SERVO_GROUND_W = 0.1   # m per share of ticks a servo housing is on the ground (rule `servo_ground`)
UNSTABLE_SCORE = -5.0
G = 9.81


def up(v: float, m: Metrics) -> float:
    """Progress scaled by how upright the torso stayed; losses stay whole."""
    return v * m.upright if v > 0 else v


def rule_cost(m: Metrics) -> float:
    """Penalty of the active optional rules (0 when none is on)."""
    c = 0.0
    if rules.active("tilt"):
        c += TILT_W * m.tilt
    if rules.active("slip"):
        c += SLIP_W * m.slip
    if rules.active("servo_ground"):
        c += SERVO_GROUND_W * m.servo_ground
    return c


def score(m: Metrics, task: str = "forward") -> float:
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}', expected one of {TASKS}")
    if m.unstable:
        return UNSTABLE_SCORE
    cost = ENERGY_W * m.energy / (m.mass * G) + rule_cost(m)
    if task == "turn":
        s = up(abs(m.yaw), m) - TURN_DRIFT_W * math.hypot(m.dx, m.dy) - cost
    else:
        s = up(m.progress, m) - LATERAL_W * abs(m.dy) - HEADING_W * abs(m.yaw) - cost
    if m.fell:
        s -= FALL_PENALTY
    return float(s)


def commands(task: str):
    """Steering commands every evaluation runs, in order."""
    return COMMANDS.get(task, (0.0,))


def task_score(ms: Sequence[Metrics], task: str = "forward") -> float:
    """Score of one terrain from the episodes run under commands(task)."""
    if task != "steer":
        return score(ms[0], task)
    if any(m.unstable for m in ms):
        return UNSTABLE_SCORE
    turns = []
    for m, c in zip(ms[1:], commands(task)[1:]):
        dist = math.hypot(m.dx, m.dy)
        if rules.active("turn_forward") and m.dx <= 0:
            dist = 0.0
        turn = min(TURN_SCALE * min(-c * m.yaw, TURN_CAP), dist)
        s = up(turn, m) - ENERGY_W * m.energy / (m.mass * G) - rule_cost(m)
        turns.append(s - FALL_PENALTY if m.fell else s)
    turn = min(turns) if rules.active("worse_turn") else np.mean(turns)
    return float(0.5 * score(ms[0], "forward") + 0.5 * turn)


def aggregate(scores: Sequence[float]) -> float:
    """
    Mean over terrains, blended with the worst one. Pure mean lets a robot
    ignore a terrain it fails on; the min term rewards robustness.
    """
    scores = np.asarray(scores, dtype=float)
    return float(0.7 * scores.mean() + 0.3 * scores.min())

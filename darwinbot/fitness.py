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
distance, one that leans 10 degrees keeps 98%. Only a full fall used to cost
anything, so a body running close to tipping over lost nothing.

energy / (m g) is in metres: the height the robot could have lifted itself
with the same work. Dividing by weight keeps heavy and light bodies on the
same footing, so the term behaves like a cost of transport.

|yaw| is the heading change over the episode in radians. Drifting sideways
alone missed a slow curve, which barely shows in a 5 s episode but takes the
robot off its line on a longer walk. HEADING_W charges the same metres per
radian that a turning episode earns (TURN_SCALE).

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
episode (then up() of that): the robot has to walk a curve, and spinning on the spot earns
nothing:

score = 0.4 * straight + 0.2 * (left + right) + 0.2 * half

A body that only walks straight, or only spins one way, falls behind one that
does both.

The half-speed episode asks for half the distance of the full-speed one. With P
the upright-weighted progress of an episode (never below 0):

half = 0.5 * P_full - |P_half - 0.5 * P_full| - energy cost - FALL_PENALTY * fell

It earns the most when the body walks at exactly half pace, nothing when it
ignores the command (P_half = P_full) or stands still, and a faster walker has
more to gain. The speed genes get no credit unless they work.

The turn credit and the turn task's drift use the whole episode (hypot(dx, dy)).

A fall ends the episode but keeps the distance covered so far, so a robot
that runs 2 m and then tips over still beats one that never moves. The
fixed penalty only breaks ties in favour of staying upright.
"""

import math
from typing import Sequence

import numpy as np

from darwinbot.sim import Metrics

LATERAL_W    = 0.5
ENERGY_W     = 0.05
FALL_PENALTY = 0.5
TURN_DRIFT_W = 0.5
TASKS        = ("forward", "turn", "steer")
SPEED_ASKED  = 0.5     # speed command of the slow episode of the steer task
COMMANDS     = {"steer": ((0.0, 1.0), (-1.0, 1.0), (1.0, 1.0), (0.0, SPEED_ASKED))}
STRAIGHT     = ((0.0, 1.0),)   # other tasks: one straight episode, full speed
TURN_SCALE   = 0.2     # m per radian of heading turned the way asked
HEADING_W    = TURN_SCALE   # m per radian of heading lost on a straight walk
TURN_CAP     = math.pi / 2   # rad: no credit for turning more than 90 degrees
UNSTABLE_SCORE = -5.0
G = 9.81


def up(v: float, m: Metrics) -> float:
    """Progress scaled by how upright the torso stayed; losses stay whole."""
    return v * m.upright if v > 0 else v


def score(m: Metrics, task: str = "forward") -> float:
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}', expected one of {TASKS}")
    if m.unstable:
        return UNSTABLE_SCORE
    cost = ENERGY_W * m.energy / (m.mass * G)
    if task == "turn":
        s = up(abs(m.yaw), m) - TURN_DRIFT_W * math.hypot(m.dx, m.dy) - cost
    else:
        s = up(m.progress, m) - LATERAL_W * abs(m.dy) - HEADING_W * abs(m.yaw) - cost
    if m.fell:
        s -= FALL_PENALTY
    return float(s)


def commands(task: str):
    """(steering, speed) commands every evaluation runs, in order."""
    return COMMANDS.get(task, STRAIGHT)


def task_score(ms: Sequence[Metrics], task: str = "forward") -> float:
    """Score of one terrain from the episodes run under commands(task)."""
    if task != "steer":
        return score(ms[0], task)
    if any(m.unstable for m in ms):
        return UNSTABLE_SCORE
    turns = []
    for m, (c, _) in zip(ms[1:3], commands(task)[1:3]):
        turn = min(TURN_SCALE * min(-c * m.yaw, TURN_CAP), math.hypot(m.dx, m.dy))
        s = up(turn, m) - ENERGY_W * m.energy / (m.mass * G)
        turns.append(s - FALL_PENALTY if m.fell else s)
    full, slow = max(up(ms[0].progress, ms[0]), 0.0), max(up(ms[3].progress, ms[3]), 0.0)
    half = (SPEED_ASKED * full - abs(slow - SPEED_ASKED * full)
            - ENERGY_W * ms[3].energy / (ms[3].mass * G))
    if ms[3].fell:
        half -= FALL_PENALTY
    return float(0.4 * score(ms[0], "forward") + 0.4 * np.mean(turns) + 0.2 * half)


def aggregate(scores: Sequence[float]) -> float:
    """
    Mean over terrains, blended with the worst one. Pure mean lets a robot
    ignore a terrain it fails on; the min term rewards robustness.
    """
    scores = np.asarray(scores, dtype=float)
    return float(0.7 * scores.mean() + 0.3 * scores.min())

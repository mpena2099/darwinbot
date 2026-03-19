"""
darwinbot.scenarios
-------------------
Each scenario module exposes a single function:

    build(client: int) -> ScenarioInfo

It loads all necessary URDF/geometry into the given PyBullet client
and returns a ScenarioInfo describing where to spawn the robot and
how to score it.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ScenarioInfo:
    name:          str
    spawn_pos:     Tuple[float, float, float] = (0.0, 0.0, 0.35)
    spawn_orn:     Tuple[float, float, float] = (0.0, 0.0, 0.0)   # euler RPY
    # Optional target position for goal-directed scenarios (e.g. confined room)
    target_pos:    Optional[Tuple[float, float, float]] = None
    # Friction override applied to scenario ground bodies (informational)
    ground_friction: float = 0.7

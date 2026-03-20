"""
darwinbot.scenarios
-------------------
Each scenario module exposes two functions:

    get_xml() -> str
        Returns MJCF XML snippet injected into the worldbody
        before model compilation (floors, walls, ramps, etc.)

    build(model, data) -> ScenarioInfo
        Called after compilation to set spawn position,
        apply friction overrides, etc.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ScenarioInfo:
    name:            str
    spawn_pos:       Tuple[float, float, float] = (0.0, 0.0, 0.35)
    spawn_orn:       Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_pos:      Optional[Tuple[float, float, float]] = None
    ground_friction: float = 1.2

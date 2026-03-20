"""
scenarios/registry.py
---------------------
Central registry of all available scenarios.

Usage
-----
from darwinbot.scenarios.registry import get_active_scenarios

scenarios = get_active_scenarios("phase2")
for name, get_xml_fn, build_fn, weight in scenarios:
    ...
"""

from typing import List, Tuple, Callable
from darwinbot.scenarios import ScenarioInfo
from darwinbot.scenarios import flat, surfaces, obstacles, confined

# (name, get_xml_fn, build_fn, weight)
REGISTRY: List[Tuple[str, Callable, Callable, float]] = [
    ("flat",      flat.get_xml,      flat.build,      1.0),
    ("surfaces",  surfaces.get_xml,  surfaces.build,  1.0),
    ("obstacles", obstacles.get_xml, obstacles.build, 1.5),
    ("confined",  confined.get_xml,  confined.build,  1.0),
]

_BY_NAME = {name: (gx, bf, w) for name, gx, bf, w in REGISTRY}

PHASE_PRESETS = {
    "phase1": ["flat"],
    "phase2": ["flat", "surfaces"],
    "phase3": ["flat", "surfaces", "obstacles"],
    "phase4": ["flat", "surfaces", "obstacles", "confined"],
    "all":    ["flat", "surfaces", "obstacles", "confined"],
}


def get_active_scenarios(preset: str = "phase1") \
        -> List[Tuple[str, Callable, Callable, float]]:
    """
    Return list of (name, get_xml_fn, build_fn, normalised_weight).
    """
    if preset in PHASE_PRESETS:
        names = PHASE_PRESETS[preset]
    else:
        names = [n.strip() for n in preset.split(",")]

    result = []
    for name in names:
        if name not in _BY_NAME:
            raise ValueError(
                f"Unknown scenario '{name}'. "
                f"Available: {list(_BY_NAME.keys())}"
            )
        gx, bf, w = _BY_NAME[name]
        result.append((name, gx, bf, w))

    total  = sum(w for _, _, _, w in result)
    result = [(n, gx, bf, w / total) for n, gx, bf, w in result]
    return result


def list_scenarios() -> List[str]:
    return [name for name, _, _, _ in REGISTRY]

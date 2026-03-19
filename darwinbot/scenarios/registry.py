"""
scenarios/registry.py
---------------------
Central registry of all available scenarios.

Usage
-----
from darwinbot.scenarios.registry import get_active_scenarios, REGISTRY

# Get the currently active scenario builders and their weights
scenarios = get_active_scenarios("phase2")

# Iterate:
for name, builder, weight in scenarios:
    info = builder(client)
    ...

Phases
------
phase1  : flat only                         (baseline)
phase2  : flat + surfaces
phase3  : flat + surfaces + obstacles
phase4  : flat + surfaces + obstacles + confined
all     : same as phase4

You can also pass a comma-separated list of scenario names:
    get_active_scenarios("flat,obstacles,confined")
"""

from typing import List, Tuple, Callable
from darwinbot.scenarios import ScenarioInfo

# Import scenario builders
from darwinbot.scenarios import flat, surfaces, obstacles, confined

# ── Registry ──────────────────────────────────────────────────────────────────
# Each entry: (name, builder_function, weight_in_composite_fitness)
#
# Weight meaning: the final fitness is a weighted average of all active
# scenario scores.  Weights do NOT need to sum to 1 — they are normalised
# automatically in fitness.py.
#
# Rationale for weights:
#   flat      : reference baseline, moderate weight
#   surfaces  : important for real environments, same weight
#   obstacles : hardest, up-weighted so it is not drowned out
#   confined  : different fitness logic, moderate weight

REGISTRY: List[Tuple[str, Callable, float]] = [
    ("flat",      flat.build,      1.0),
    ("surfaces",  surfaces.build,  1.0),
    ("obstacles", obstacles.build, 1.5),
    ("confined",  confined.build,  1.0),
]

_BY_NAME = {name: (builder, weight) for name, builder, weight in REGISTRY}

PHASE_PRESETS = {
    "phase1": ["flat"],
    "phase2": ["flat", "surfaces"],
    "phase3": ["flat", "surfaces", "obstacles"],
    "phase4": ["flat", "surfaces", "obstacles", "confined"],
    "all":    ["flat", "surfaces", "obstacles", "confined"],
}


def get_active_scenarios(preset: str = "phase1") \
        -> List[Tuple[str, Callable, float]]:
    """
    Return list of (name, builder, weight) for the given preset or
    comma-separated list of scenario names.

    Parameters
    ----------
    preset : str
        One of the PHASE_PRESETS keys, or a comma-separated list of
        scenario names from REGISTRY (e.g. "flat,obstacles").

    Returns
    -------
    List of (name, builder_fn, weight) tuples, normalised so weights sum to 1.
    """
    if preset in PHASE_PRESETS:
        names = PHASE_PRESETS[preset]
    else:
        # Treat as comma-separated name list
        names = [n.strip() for n in preset.split(",")]

    result = []
    for name in names:
        if name not in _BY_NAME:
            raise ValueError(
                f"Unknown scenario '{name}'. "
                f"Available: {list(_BY_NAME.keys())}"
            )
        builder, weight = _BY_NAME[name]
        result.append((name, builder, weight))

    # Normalise weights
    total = sum(w for _, _, w in result)
    result = [(n, b, w / total) for n, b, w in result]

    return result


def list_scenarios() -> List[str]:
    """Return names of all registered scenarios."""
    return [name for name, _, _ in REGISTRY]

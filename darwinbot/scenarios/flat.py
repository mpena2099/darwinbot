"""
scenarios/flat.py
-----------------
Phase 1 — baseline flat floor.
The floor plane is already defined in the base MJCF,
so get_xml() returns an empty string.
"""

from darwinbot.scenarios import ScenarioInfo


def get_xml() -> str:
    """No extra geometry needed — base MJCF already has a floor plane."""
    return ""


def build(model, data) -> ScenarioInfo:
    return ScenarioInfo(
        name="flat",
        spawn_pos=(0.0, 0.0, 0.35),
        ground_friction=1.2,
    )

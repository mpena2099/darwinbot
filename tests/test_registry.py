"""
tests/test_registry.py
----------------------
Unit tests for the scenario registry and phase presets.
"""

import pytest
from darwinbot.scenarios.registry import (
    get_active_scenarios, list_scenarios, PHASE_PRESETS,
)


def test_list_scenarios_returns_all_four():
    names = list_scenarios()
    assert set(names) == {"flat", "surfaces", "obstacles", "confined"}


def test_phase_presets_exist():
    for preset in ("phase1", "phase2", "phase3", "phase4", "all"):
        result = get_active_scenarios(preset)
        assert len(result) > 0


def test_weights_normalise_to_one():
    for preset in PHASE_PRESETS:
        result = get_active_scenarios(preset)
        total = sum(w for _, _, _, w in result)
        assert abs(total - 1.0) < 1e-9, f"Weights for {preset} don't sum to 1.0: {total}"


def test_phase1_single_scenario():
    result = get_active_scenarios("phase1")
    assert len(result) == 1
    assert result[0][0] == "flat"


def test_phase4_all_scenarios():
    result = get_active_scenarios("phase4")
    names = [n for n, _, _, _ in result]
    assert "flat" in names
    assert "surfaces" in names
    assert "obstacles" in names
    assert "confined" in names


def test_custom_comma_list():
    result = get_active_scenarios("flat,obstacles")
    names = [n for n, _, _, _ in result]
    assert names == ["flat", "obstacles"]
    total = sum(w for _, _, _, w in result)
    assert abs(total - 1.0) < 1e-9


def test_unknown_scenario_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_active_scenarios("nonexistent")


def test_each_entry_has_callables():
    for preset in PHASE_PRESETS:
        for name, get_xml_fn, build_fn, weight in get_active_scenarios(preset):
            assert callable(get_xml_fn), f"{name}.get_xml is not callable"
            assert callable(build_fn),   f"{name}.build is not callable"
            assert weight > 0

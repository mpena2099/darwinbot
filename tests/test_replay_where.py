"""
test_replay_where.py
--------------------
The --where filter of replay.py --list: parsing and matching, no simulation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import replay  # noqa: E402

ROW = {"fitness": 0.4, "duty": 0.3, "height": 0.5, "servos": 8, "legs": 4,
       "cot": 6.0, "wheels": 0, "feet": 2, "segments": 0, "passive": 1, "nodes": 12}


@pytest.mark.parametrize("expr, op, value", [
    ("fitness=0.4", "=", 0.4), ("fitness!=0.4", "!=", 0.4),
    ("fitness<0.5", "<", 0.5), ("fitness<=0.4", "<=", 0.4),
    ("fitness>0.3", ">", 0.3), ("fitness>=0.4", ">=", 0.4),
])
def test_every_operator_parses(expr, op, value):
    assert replay.parse_where(expr) == [("fitness", op, value)]


@pytest.mark.parametrize("expr, expected", [
    ("fitness=0.4", True), ("fitness!=0.4", False),
    ("fitness<0.4", False), ("fitness<=0.4", True),
    ("fitness>0.4", False), ("fitness>=0.4", True),
])
def test_every_operator_matches(expr, expected):
    assert replay.matches(ROW, replay.parse_where(expr)) is expected


def test_multiple_conditions_all_must_hold():
    conds = replay.parse_where("wheels=0,legs>=4,fitness>0.3")
    assert replay.matches(ROW, conds)
    assert not replay.matches(ROW, replay.parse_where("wheels=0,legs>=5"))
    assert not replay.matches(ROW, replay.parse_where("wheels=1,legs>=4"))


def test_spaces_around_tokens_are_ignored():
    assert replay.parse_where(" wheels = 0 , legs >= 4 ") == [
        ("wheels", "=", 0.0), ("legs", ">=", 4.0)]


def test_matches_true_and_false():
    assert replay.matches(ROW, replay.parse_where("feet=2,passive>=1"))
    assert not replay.matches(ROW, replay.parse_where("feet=2,passive>1"))


def test_unknown_field_names_the_valid_ones():
    with pytest.raises(SystemExit, match="unknown field 'speed'.*fitness.*nodes"):
        replay.parse_where("speed>1")


def test_missing_operator_is_an_error():
    with pytest.raises(SystemExit, match="bad condition 'legs 4'"):
        replay.parse_where("legs 4")


def test_non_number_is_an_error():
    with pytest.raises(SystemExit, match="'four' .* is not a number"):
        replay.parse_where("legs>=four")
    with pytest.raises(SystemExit, match="is not a number"):
        replay.parse_where("legs>=")

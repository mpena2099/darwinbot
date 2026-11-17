"""
test_replay_watch.py
--------------------
The reload logic of replay.py --robot --watch, without a display.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import replay  # noqa: E402
from darwinbot.terrain import get_terrains  # noqa: E402

GOOD = """name: t
torso: {length_mm: 200, width_mm: 140}
nodes:
  hip:  {parent: torso, slot: 2, kind: joint, axis: pitch, length_mm: %d, roll_deg: 270}
  foot: {parent: hip, slot: 0, kind: foot, radius_mm: 15, shape: sphere}
"""
TERRAIN = get_terrains("flat")[0]


def _write(path, text, mtime):
    path.write_text(text)
    os.utime(path, (mtime, mtime))


def test_reload_cycle(tmp_path):
    f = tmp_path / "r.yaml"
    _write(f, GOOD % 80, 1000)
    state, msgs, changed = replay.reload_robot(f, None, TERRAIN)
    assert changed and state is not None and msgs

    # same mtime: nothing to do
    same, msgs, changed = replay.reload_robot(f, state, TERRAIN)
    assert same is state and msgs == [] and not changed

    # a new valid body replaces the old one
    _write(f, GOOD % 100, 2000)
    new, msgs, changed = replay.reload_robot(f, state, TERRAIN)
    assert changed and new is not state and new.mtime == 2000
    assert new.morph is not state.morph and msgs

    # a file that does not parse keeps the last good body and says why, once
    _write(f, "torso: [", 3000)
    kept, msgs, changed = replay.reload_robot(f, new, TERRAIN)
    assert not changed and kept.episode is new.episode and kept.mtime == 3000
    assert msgs and "not valid YAML" in msgs[0]
    again, msgs, _ = replay.reload_robot(f, kept, TERRAIN)
    assert msgs == []

    # a rule error (a thin link that would break under the servo) keeps it too
    _write(f, (GOOD % 200).replace("roll_deg: 270", "roll_deg: 270, width_mm: 12, thick_mm: 3", 1), 4000)
    kept2, msgs, changed = replay.reload_robot(f, kept, TERRAIN)
    assert not changed and kept2.episode is new.episode
    assert msgs and any("keeping the last good body" in m for m in msgs)


def test_first_load_errors(tmp_path):
    f = tmp_path / "r.yaml"
    _write(f, "torso: [", 1000)
    state, msgs, changed = replay.reload_robot(f, None, TERRAIN)
    assert state is None and not changed and msgs

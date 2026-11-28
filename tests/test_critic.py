"""experiments/critic.py on a tiny archive: a rolling cart and one with a passive tail."""

import sys

import numpy as np

sys.path.insert(0, "experiments")

from darwinbot import controller as ctl   # noqa: E402
from darwinbot import evolve              # noqa: E402
from darwinbot import morphology as mo    # noqa: E402
import critic                             # noqa: E402

FIELDS = {"type", "servos", "nodes", "fitness", "progress", "useless",
          "useless_limbs", "turn_ratio", "body_ground", "servo_ground",
          "tilt", "slip", "upright", "defects"}


def _archive():
    """Two cells: the same rolling gait with and without a tail on the top face."""
    cart = mo.decode(mo.encode(mo.template(1, wheels=True, yaw=True, wheel_rad=0.06)))
    tailed = cart.copy()
    slot = max(tailed.nodes) + 1
    tailed.nodes[slot] = mo.Node(slot, "tail", 0, 8, length=0.04)
    ctrl = np.full(ctl.CTRL_LENGTH, 0.5)
    for act in cart.layout():
        if act.kind == "wheel":
            ctrl[ctl.node_base(act.slot) + ctl.OFFSET] = 0.3
    a = evolve.Archive(terrains="flat", seconds=1.0, variants=0, task="forward")
    for genes, behaviour in ((mo.encode(cart), [0.9, 0.3, 8, 0, 1, 4]),
                             (mo.encode(tailed), [0.3, 0.3, 8, 0, 1, 4])):
        assert a.insert(evolve.Elite(genes, ctrl, 0.1, [0.1], 0, behaviour))
    return a, slot


def test_every_robot_has_a_full_row_and_the_tail_is_useless():
    a, tail_slot = _archive()
    names = set(a.names().values())
    rows, counts = critic.critic(a, workers=1)
    assert len(rows) == len(names)
    assert {r["type"] for r in rows} == names
    for row in rows:
        assert FIELDS <= set(row)
        assert np.isfinite([row["fitness"], row["progress"], row["turn_ratio"],
                            row["body_ground"], row["servo_ground"], row["tilt"],
                            row["slip"], row["upright"]]).all()
        assert row["useless"] == len(row["useless_limbs"])
        assert set(row["defects"]) <= set(critic.DEFECTS)
        assert all(set(limb) == {"slot", "kind", "servos", "drop"}
                   for limb in row["useless_limbs"])
    tailed = next(r for r in rows if r["nodes"] == tail_slot + 1)
    assert tailed["useless"] >= 1
    assert any(limb["slot"] == tail_slot and limb["kind"] == "tail"
               and limb["servos"] == 0 and limb["drop"] < critic.USELESS_DROP
               for limb in tailed["useless_limbs"])


def test_defect_counts_sum_the_rows():
    a, _ = _archive()
    rows, counts = critic.critic(a, workers=1)
    assert counts == {d: sum(d in r["defects"] for r in rows)
                      for d in critic.DEFECTS}


def test_cli_writes_the_same_data(tmp_path):
    a, _ = _archive()
    path = tmp_path / "archive.pkl"
    import pickle
    with open(path, "wb") as f:
        pickle.dump(a, f)
    out = tmp_path / "report.json"
    import json
    import subprocess
    r = subprocess.run([sys.executable, "experiments/critic.py", "--archive", str(path),
                        "--top", "1", "--json", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert len(data["rows"]) == 1
    assert data["counts"] == critic.counts(data["rows"])
    assert "defects:" in r.stdout

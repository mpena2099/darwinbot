"""
test_migrate.py
---------------
Archives of an older BODY_MODEL: refused by a plain load, translated and
simulated again by experiments/migrate_archive.py.
"""

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from darwinbot import evolve, hardware
from darwinbot import controller as co

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import migrate_archive as ma   # noqa: E402

OLD = evolve.BODY_MODEL - 1


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return evolve.run(n_init=3, n_batches=0, batch_size=2, budget_init=4,
                      workers=1, out_dir=str(tmp_path_factory.mktemp("tiny")),
                      variants=0)


def _stale_file(archive, tmp_path, version=OLD):
    old = pickle.loads(pickle.dumps(archive))
    old.body_model = version
    path = tmp_path / "old.pkl"
    with open(path, "wb") as f:
        pickle.dump(old, f)
    return path


def _load(path):
    with open(path, "rb") as f, evolve.loading_stale():
        return pickle.load(f)


def test_migration_path():
    assert evolve.migration_path(evolve.BODY_MODEL) == []
    assert evolve.migration_path(22) == list(range(22, evolve.BODY_MODEL))
    assert evolve.migration_path(1) is None
    assert evolve.migration_path(evolve.BODY_MODEL + 1) is None
    assert evolve.migration_path(None) is None


def test_stale_archive_is_refused_with_the_migration_hint(tiny, tmp_path, monkeypatch):
    monkeypatch.setitem(evolve.MIGRATIONS, OLD, lambda m, c: (m, c))
    path = _stale_file(tiny, tmp_path)
    with pytest.raises(ValueError, match="migrate_archive.py"), open(path, "rb") as f:
        pickle.load(f)
    assert _load(path).body_model == OLD


def test_stale_archive_without_a_path_is_refused_even_when_loading_stale(tiny, tmp_path):
    path = _stale_file(tiny, tmp_path, version=1)
    with pytest.raises(ValueError, match="retrain"), open(path, "rb") as f:
        pickle.load(f)
    assert evolve.migration_path(_load(path).body_model) is None


def test_migrate_keeps_settings_and_adds_history(tiny, tmp_path, monkeypatch):
    monkeypatch.setitem(evolve.MIGRATIONS, OLD, lambda m, c: (m, c))
    old = _load(_stale_file(tiny, tmp_path))
    old.activate()
    out, summary = ma.migrate(old, workers=1, budget=4)
    assert out.body_model == evolve.BODY_MODEL
    for k in ("terrains", "seconds", "variants", "task", "payload", "ablate",
              "rules", "body", "profile", "locked", "plugin"):
        assert getattr(out, k) == getattr(tiny, k)
    assert len(out.history) == len(tiny.history) + 1
    assert out.history[-1]["migrated_from"] == OLD
    assert summary["elites"] == len(tiny.cells) and summary["dropped"] == 0
    assert summary["cells"] == len(out.cells)
    # every old elite is a parent in the family tree or kept its record
    for uid in tiny.lineage:
        assert uid in out.lineage


def test_unbuildable_elite_is_dropped(tiny, tmp_path, monkeypatch):
    monkeypatch.setitem(evolve.MIGRATIONS, OLD, lambda m, c: (m, c))
    old = _load(_stale_file(tiny, tmp_path))
    old.activate()
    first = [True]
    real = ma.optimise_controller

    def fake(*args, **kwargs):
        res = real(*args, **kwargs)
        if first[0]:
            first[0] = False
            res.evals = 0
        return res

    monkeypatch.setattr(ma, "optimise_controller", fake)
    n = len(old.cells)
    if n < 2:
        pytest.skip("tiny archive has one elite")
    out, summary = ma.migrate(old, workers=1, budget=4)
    assert summary["dropped"] == 1 and summary["elites"] == n


def test_gene_function_is_applied_and_recorded_in_lineage(tiny, tmp_path, monkeypatch):
    seen = []

    def shift(morph, ctrl):
        seen.append(1)
        c = np.array(ctrl, dtype=float)
        c[0] += 0.01
        return morph, c

    monkeypatch.setitem(evolve.MIGRATIONS, OLD, shift)
    old = _load(_stale_file(tiny, tmp_path))
    old.activate()
    out, _ = ma.migrate(old, workers=1, budget=4)
    assert len(seen) == len(tiny.cells)
    old_ids = {evolve.elite_id(e.morph, e.ctrl) for e in tiny.cells.values()}
    parents = {out.lineage[evolve.elite_id(e.morph, e.ctrl)][0] for e in out.cells.values()}
    assert parents <= old_ids | {None}
    assert parents & old_ids


def test_profile_choice():
    current = hardware.default_profile()
    old = copy.deepcopy(current)
    old["servo"]["bracket_mass"] = 0.015          # keys of another version
    del old["servo"]["cradle_mass_plate"]
    assert ma.is_default_profile(old)
    assert ma.choose_profile(old, "auto") == (current, None)
    custom = copy.deepcopy(current)
    custom["battery"]["volts"] = 7.4
    assert not ma.is_default_profile(custom)
    kept, note = ma.choose_profile(custom, "auto")
    assert kept == custom and note
    assert ma.choose_profile(custom, "default")[0] == current
    assert ma.choose_profile(old, "keep")[0] == old
    with pytest.raises(SystemExit):
        ma.check_keys(old)
    ma.check_keys(current)

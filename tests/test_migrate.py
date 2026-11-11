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
    assert evolve.migration_path(20) == [20, 21, 22]
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


def test_seconds_overrides_the_episode_length(tiny, tmp_path, monkeypatch):
    monkeypatch.setitem(evolve.MIGRATIONS, OLD, lambda m, c: (m, c))
    old = _load(_stale_file(tiny, tmp_path))
    old.activate()
    used = []
    real = ma.optimise_controller

    def spy(*args, **kwargs):
        used.append(kwargs["seconds"])
        return real(*args, **kwargs)

    monkeypatch.setattr(ma, "optimise_controller", spy)
    new_seconds = tiny.seconds + 1.0
    out, summary = ma.migrate(old, workers=1, budget=4, seconds=new_seconds)
    assert out.seconds == new_seconds and set(used) == {new_seconds}
    assert summary["seconds_before"] == tiny.seconds
    assert summary["seconds_after"] == new_seconds


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


# ---- BODY_MODEL 20 and 21 ---------------------------------------------------

from darwinbot import morphology as mo   # noqa: E402


def _wheeled_genes():
    rng = np.random.default_rng(3)
    for _ in range(200):
        genes = mo.random_morphology(rng)
        if any(n.kind == "wheel" for n in mo.decode(genes).nodes.values()):
            return genes
    pytest.skip("no random wheeled body found")


def test_wheel_radius_survives_20_and_width_is_the_old_one():
    genes = _wheeled_genes()
    blocks = genes.reshape(mo.MAX_NODES, mo.BLOCK_GENES).copy()
    wheels = [i for i, n in mo.decode(genes).nodes.items() if n.kind == "wheel"]
    # a genome as body model 20 stored it: the radius gene in the 30-60 mm range
    rad_m = {}
    for i in wheels:
        rad_m[i] = round(0.03 + 0.37 * 0.03 + 0.01 * i / 20, 6)
        blocks[i, mo.WHEEL] = (rad_m[i] - 0.03) / 0.03
    old = blocks.reshape(-1)
    ctrl = np.random.default_rng(0).uniform(size=1 + mo.MAX_NODES * 8)
    genes = old
    for step in evolve.migration_path(20):
        genes, ctrl = evolve.MIGRATIONS[step](genes, ctrl)
    new = mo.decode(genes)
    for i in wheels:
        assert new.nodes[i].wheel_rad == pytest.approx(rad_m[i], abs=2e-6)
        assert new.nodes[i].wheel_width == pytest.approx(max(0.02, 0.5 * rad_m[i]), abs=2e-6)
    assert len(genes) == mo.MORPH_LENGTH and len(ctrl) == co.CTRL_LENGTH


def test_controller_padding_is_neutral_and_keeps_the_old_genes():
    old = np.random.default_rng(1).uniform(size=1 + mo.MAX_NODES * 8)
    _, new = evolve.MIGRATIONS[21](_wheeled_genes(), old)
    assert new.shape == (co.CTRL_LENGTH,) and new[0] == old[0]
    n_old = old[1:].reshape(mo.MAX_NODES, 8)
    n_new = new[1:].reshape(mo.MAX_NODES, co.CTRL_PER_NODE)
    assert np.array_equal(n_new[:, :8], n_old)
    assert np.all(n_new[:, co.SPEED_GAIN:] == 0.5)


def test_padded_controller_gives_the_same_targets_at_any_speed():
    genes = mo.random_morphology(np.random.default_rng(5))
    layout = mo.decode(genes).layout()
    old = np.random.default_rng(2).uniform(size=1 + mo.MAX_NODES * 8)
    _, new = evolve.MIGRATIONS[21](genes, old)
    full = co.Params(new, layout, 0.0, 1.0)
    for speed in (0.0, 0.4):
        slow = co.Params(new, layout, 0.0, speed)
        for k in ("amp1", "amp2", "offset", "phase1", "phase2", "sigma"):
            assert np.allclose(getattr(full, k), getattr(slow, k)), k


def _body_model_20_file(archive, tmp_path):
    """A pickle shaped like a showcase-1 archive: a four-axis grid, controllers
    of 8 genes per slot, and none of the settings added since."""
    from ribs.archives import CVTArchive
    grid = CVTArchive(solution_dim=mo.MORPH_LENGTH, centroids=60,
                      ranges=[(0.0, 1.0)] * 4,
                      extra_fields={"ctrl": ((1 + mo.MAX_NODES * 8,), np.float64),
                                    "scores": ((len(archive.cells[next(iter(archive.cells))].scores),),
                                               np.float64),
                                    "born": ((), np.int64), "behaviour": ((6,), np.float64)})
    for e in archive.cells.values():
        slots = e.ctrl[1:].reshape(mo.MAX_NODES, co.CTRL_PER_NODE)[:, :8]
        old_ctrl = np.concatenate([e.ctrl[:1], slots.reshape(-1)])
        grid.add(solution=[e.morph], objective=[e.fitness],
                 measures=[evolve.normalise(e.behaviour)[[0, 1, 3, 4]]],
                 ctrl=[old_ctrl], scores=[e.scores], born=[e.born], behaviour=[e.behaviour])
    old = pickle.loads(pickle.dumps(archive))
    old._grid = grid
    old.body_model = 20
    for k in ("profile", "plugin", "locked", "body", "rules", "ablate", "payload", "task",
              "variants", "lineage"):
        del old.__dict__[k]
    path = tmp_path / "sc.pkl"
    with open(path, "wb") as f:
        pickle.dump(old, f)
    return path


def test_showcase_shaped_archive_loads_and_migrates(tiny, tmp_path):
    path = _body_model_20_file(tiny, tmp_path)
    with pytest.raises(ValueError, match="migrate_archive.py"), open(path, "rb") as f:
        pickle.load(f)
    old = _load(path)
    assert old.body_model == 20 and len(old.cells) == len(tiny.cells)
    assert old.profile == hardware.default_profile()
    assert old.task == "forward" and old.plugin == {} and old.lineage == {}
    old.activate()
    out, summary = ma.migrate(old, workers=1, budget=4)
    assert out.body_model == evolve.BODY_MODEL and summary["from"] == 20
    assert summary["dropped"] == 0 and len(out.cells) >= 1
    assert all(len(e.ctrl) == co.CTRL_LENGTH for e in out.cells.values())

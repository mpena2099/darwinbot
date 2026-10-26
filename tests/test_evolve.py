"""
test_evolve.py
--------------
MAP-Elites cells, archive replacement and a tiny end to end run.
"""

import pickle

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo


GENES = np.full(mo.MORPH_LENGTH, 0.5)
CTRL = np.zeros(co.CTRL_LENGTH)


def _elite(fitness, behaviour, genes=GENES, born=0):
    return evolve.Elite(genes, CTRL, fitness, [fitness], born,
                        np.array(behaviour, dtype=float))


def test_cell_of_is_deterministic_and_in_range():
    cell = evolve.cell_of([0.5, 0.7, 8, 0.0, 1.0, 2])
    assert cell == evolve.cell_of([0.5, 0.7, 8, 0.0, 1.0, 2])
    assert 0 <= cell < evolve.N_CELLS
    assert 0 <= evolve.cell_of([5.0, 9.0, 99, 0.0, 1.0, 2]) < evolve.N_CELLS   # clipped


def test_insert_keeps_best_per_cell():
    behaviour = [0.9, 0.3, 8, 0.0, 1.0, 2]
    cell = evolve.cell_of(behaviour)
    archive = evolve.Archive(terrains="flat")

    assert archive.insert(_elite(1.0, behaviour))
    assert not archive.insert(_elite(0.5, behaviour))
    assert archive.cells[cell].fitness == 1.0
    assert archive.insert(_elite(2.0, behaviour))
    assert archive.best().fitness == 2.0
    assert len(archive.cells) == 1
    # the behaviour vector survives the round trip through the ribs archive
    assert np.allclose(archive.cells[cell].behaviour, behaviour)


def test_run_writes_archive(tmp_path):
    archive = evolve.run(n_init=2, n_batches=1, batch_size=2,
                         budget_init=6, budget_child=4, workers=1,
                         out_dir=str(tmp_path))

    path = tmp_path / "archive.pkl"
    assert path.exists()
    assert archive.cells
    assert all(len(e.behaviour) == 6 for e in archive.cells.values())
    assert len(archive.history) == 2
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, evolve.Archive)
    # every elite has a lineage record; children point to an earlier elite
    assert loaded.lineage
    for e in loaded.cells.values():
        chain = loaded.ancestors(e)
        assert chain and chain[-1][1][0] is None
        assert all(rec[1] <= e.born for _, rec in chain)


def test_lineage_survives_parent_eviction():
    a = evolve.Archive(terrains="flat")
    rng = np.random.default_rng(0)
    morph = mo.encode(mo.template(2))
    beh = np.array([0.6, 0.3, 8.0, 0.0, 1.0, 2.0])
    old = evolve.Elite(morph, rng.random(evolve.ctrl.CTRL_LENGTH), 0.1, [0.1], 0, beh)
    new = evolve.Elite(morph, rng.random(evolve.ctrl.CTRL_LENGTH), 0.5, [0.5], 1, beh)
    pid = evolve.elite_id(old.morph, old.ctrl)
    assert a.insert(old)
    assert a.insert(new, parent=pid)          # same cell: the parent is evicted
    assert len(a.cells) == 1
    chain = a.ancestors(next(iter(a.cells.values())))
    assert [uid for uid, _ in chain][1] == pid and chain[1][1][0] is None


def test_type_name_bins():
    assert evolve.type_name([0.9, 0.2, 8, 0.0]) == "8servos-low-crawl"
    assert evolve.type_name([0.3, 0.2, 8, 0.0]) == "8servos-low-trot"
    assert evolve.type_name([0.05, 0.2, 8, 0.0]) == "8servos-low-belly"
    assert evolve.type_name([0.6, 0.5, 12, 0.0]) == "12servos-mid-walk"
    assert evolve.type_name([0.3, 1.0, 16, 0.0]) == "16servos-high-trot"
    assert evolve.type_name([0.9, 0.2, 4, 0.8]) == "4servos-low-roll"   # wheels win


def test_names_round_trip_with_parse():
    archive = evolve.Archive(terrains="flat")
    rng = np.random.default_rng(3)
    for i in range(40):
        archive.insert(_elite(float(i), [rng.uniform(), rng.uniform(0, 1.5),
                                         int(rng.integers(4, 17)), rng.choice([0.0, 1.0]),
                                         rng.uniform(0.1, 100), int(rng.integers(0, 9))]))
    names = archive.names()
    assert len(set(names.values())) == len(names)          # unique
    for cell, name in names.items():
        assert archive.find(name) == cell
    shared = [n for n in names.values() if n.rsplit("-", 1)[-1].isdigit()
              and evolve._NAME.match(n).group(4)]
    assert shared, "40 elites in few name bins should collide at least once"


def _at(i):
    """Behaviour vector sitting on the centroid of cell i."""
    duty, height, legs, cot = evolve._centroids()[i]
    return [duty, height * evolve.HEIGHT_MAX, 8, 0.0,
            evolve.COT_MIN * (evolve.COT_MAX / evolve.COT_MIN) ** cot, legs * mo.MAX_LEGS]


def test_name_collision_appends_cell_index():
    by_name = {}
    for i in range(evolve.N_CELLS):
        by_name.setdefault(evolve.type_name(_at(i)), []).append(i)
    name, (i, j) = next((n, c[:2]) for n, c in by_name.items() if len(c) > 1)
    archive = evolve.Archive(terrains="flat")
    archive.insert(_elite(1.0, _at(i)))
    archive.insert(_elite(2.0, _at(j)))
    names = archive.names()
    assert set(names.values()) == {f"{name}-{evolve.cell_of(_at(i))}",
                                   f"{name}-{evolve.cell_of(_at(j))}"}
    assert archive.find(names[evolve.cell_of(_at(j))]) == evolve.cell_of(_at(j))
    with pytest.raises(ValueError, match="ambiguous"):
        archive.find(name)


def test_parse_rejects_bad_names():
    assert evolve.parse_cell_name("8servos-low-crawl") == ("8servos-low-crawl", None)
    assert evolve.parse_cell_name("8servos-low-crawl-17") == ("8servos-low-crawl", 17)
    for bad in ("6legs-2wheels-14servos", "8servos-tiny-crawl", "x"):
        with pytest.raises(ValueError):
            evolve.parse_cell_name(bad)


def test_novelty_selection_prefers_isolated_elites():
    centroids = evolve._centroids()
    corner = np.ones(evolve.N_DESCRIPTORS)
    order = np.argsort(np.linalg.norm(centroids - corner, axis=1))
    archive = evolve.Archive(terrains="flat")
    for i in order[:30]:                       # a dense cluster in one corner ...
        archive.insert(_elite(1.0, _at(int(i))))
    loner = int(order[-1])                     # ... and one elite on the far side
    archive.insert(_elite(1.0, _at(loner)))

    data = archive._grid.data()
    loner_row = int(np.flatnonzero(data["index"] == loner)[0])
    nov = archive.novelty()
    assert nov[loner_row] == nov.max()

    n = 3000
    hits = lambda mode: np.mean(archive.sample_elites(
        n, np.random.default_rng(0), mode)["index"] == loner)
    assert hits("uniform") < 0.06                    # about 1 / 31
    assert hits("novelty") > 3 * hits("uniform")


def test_ctrl_c_stops_cleanly_and_keeps_the_archive(tmp_path, monkeypatch, capsys):
    calls = []
    real = evolve.Archive.stats

    def interrupted(self):
        calls.append(1)
        if len(calls) == 2:           # second batch: the user presses Ctrl+C
            raise KeyboardInterrupt
        return real(self)

    monkeypatch.setattr(evolve.Archive, "stats", interrupted)
    archive = evolve.run(n_init=2, n_batches=3, batch_size=2, budget_init=4,
                         budget_child=4, workers=1, episode_seconds=0.5,
                         variants=0, out_dir=str(tmp_path))

    assert archive.cells
    with open(tmp_path / "archive.pkl", "rb") as f:
        assert isinstance(pickle.load(f), evolve.Archive)
    out = capsys.readouterr().out
    assert "stopped, archive saved at batch 1" in out
    assert "Traceback" not in out


def test_run_logs_novelty_and_uniform_selection(tmp_path):
    archive = evolve.run(n_init=3, n_batches=1, batch_size=2, budget_init=4,
                         budget_child=4, workers=1, episode_seconds=0.5,
                         variants=0, out_dir=str(tmp_path),
                         parent_selection="uniform")
    assert all("novelty" in h for h in archive.history)


def test_grid_archive_from_before_cvt_is_rejected():
    from ribs.archives import GridArchive
    state = dict(evolve.Archive(terrains="flat").__dict__)
    state["_grid"] = GridArchive(solution_dim=1, dims=(2,), ranges=[(0, 1)])
    with pytest.raises(ValueError, match="older body model"):
        evolve.Archive.__new__(evolve.Archive).__setstate__(state)


def test_old_archive_is_rejected():
    archive = evolve.Archive(terrains="flat")
    state = dict(archive.__dict__)
    state.pop("body_model")
    with pytest.raises(ValueError, match="older body model"):
        evolve.Archive.__new__(evolve.Archive).__setstate__(state)


def test_variants_deterministic_and_per_terrain():
    from darwinbot.inner import optimise_controller
    from darwinbot.terrain import get_terrains
    morph = mo.decode(np.full(mo.MORPH_LENGTH, 0.5))
    terrains = get_terrains("terrain")
    kw = dict(budget=4, seed=5, seconds=1.0, variants=2)
    a = optimise_controller(morph, terrains, **kw)
    b = optimise_controller(morph, terrains, **kw)
    assert a.fitness == b.fitness
    assert np.array_equal(a.ctrl, b.ctrl)
    assert len(a.scores) == len(terrains)


def test_gpu_rejects_variants():
    from darwinbot.inner import optimise_controller
    from darwinbot.terrain import get_terrains
    morph = mo.decode(np.full(mo.MORPH_LENGTH, 0.5))
    with pytest.raises(ValueError, match="CPU only"):
        optimise_controller(morph, get_terrains("flat"), device="gpu", variants=2)


def test_validate_runs_on_tiny_archive(tmp_path):
    import subprocess
    import sys
    evolve.run(n_init=2, n_batches=0, batch_size=2, budget_init=6,
               workers=1, episode_seconds=1.0, out_dir=str(tmp_path))
    r = subprocess.run([sys.executable, "experiments/validate.py", "--archive",
                        str(tmp_path / "archive.pkl"), "--n", "2"],
                       capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    assert "mean loss" in r.stdout


def test_lineage_tool_orders_ancestors_oldest_first():
    import importlib.util
    spec = importlib.util.spec_from_file_location("lineage", "experiments/lineage.py")
    lineage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lineage)

    a = evolve.Archive(terrains="flat")
    rng = np.random.default_rng(1)
    morph = mo.encode(mo.template(2))
    pid = None
    for batch, beh in enumerate(([0.9, 0.3, 8.0, 0.0, 1.0, 2.0], [0.6, 0.3, 8.0, 0.0, 1.0, 2.0],
                                 [0.3, 0.3, 8.0, 0.0, 1.0, 2.0])):
        e = evolve.Elite(morph, rng.random(evolve.ctrl.CTRL_LENGTH), 0.1 * batch,
                         [0.1 * batch], batch, np.array(beh))
        assert a.insert(e, parent=pid)
        pid = evolve.elite_id(e.morph, e.ctrl)
    steps = lineage.chain(a, e)
    assert [b for b, _, _ in steps] == [0, 1, 2]
    assert [t.split("-")[-1] for _, _, t in steps] == ["crawl", "walk", "trot"]
    rows = lineage.overview(a)
    assert rows[:, 1].max() == 3


def test_archive_task_and_payload_roundtrip_and_old_pickles_load():
    a = evolve.Archive(terrains="flat", task="turn", payload=0.3)
    b = pickle.loads(pickle.dumps(a))
    assert (b.task, b.payload) == ("turn", 0.3)

    state = dict(a.__dict__)                  # an archive saved before the fields
    del state["task"], state["payload"]
    old = evolve.Archive.__new__(evolve.Archive)
    old.__setstate__(state)
    assert (old.task, old.payload) == ("forward", 0.0)


def test_run_with_turn_task(tmp_path):
    archive = evolve.run(n_init=2, n_batches=1, batch_size=2, budget_init=4,
                         budget_child=4, workers=1, episode_seconds=1.0,
                         variants=0, task="turn", payload=0.1,
                         out_dir=str(tmp_path))
    assert archive.cells
    assert (archive.task, archive.payload) == ("turn", 0.1)
    with pytest.raises(ValueError):
        evolve.run(task="fly", out_dir=str(tmp_path))


def test_train_presets_run_in_sequence_and_resume(tmp_path):
    import subprocess
    import sys
    cmd = [sys.executable, "experiments/train.py", "--preset", "turn,step",
           "--init", "2", "--batches", "1", "--batch-size", "2",
           "--budget-init", "4", "--budget-child", "4", "--workers", "1",
           "--variants", "0", "--seconds", "1", "--out", str(tmp_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for name, task in (("turn", "turn"), ("step", "forward")):
        with open(tmp_path / name / "archive.pkl", "rb") as f:
            assert pickle.load(f).task == task
    again = subprocess.run(cmd[:4] + cmd[6:], capture_output=True, text=True)
    assert "resuming" in again.stdout


def test_insert_records_no_self_loop_in_lineage():
    a = evolve.Archive(terrains="flat")
    e = _elite(1.0, [0.6, 0.3, 8.0, 0.0, 1.0, 2.0])
    uid = evolve.elite_id(e.morph, e.ctrl)
    assert a.insert(e, parent=uid)
    assert a.lineage[uid][0] is None
    assert len(a.ancestors(e)) == 1


def test_unbuildable_results_never_enter_the_archive(monkeypatch, tmp_path):
    """Every body of the first batch is unbuildable: the run draws new random
    bodies until one works, and no evals == 0 result is inserted."""
    calls = {"n": 0}

    def fake_work(job):
        calls["n"] += 1
        genes = job[0]
        if calls["n"] <= 3:                       # the whole first batch
            return (genes, CTRL, -5.0, [-5.0], 0, np.zeros(5), job[10])
        return (genes, CTRL, 1.0, [1.0], 3,
                np.array([0.6, 0.3, 8.0, 0.0, 1.0, 2.0]), job[10])

    class FakePool:
        def __init__(self, *a): pass
        def imap_unordered(self, f, tasks): return map(f, tasks)
        def close(self): pass
        def join(self): pass

    monkeypatch.setattr(evolve, "_work", fake_work)
    monkeypatch.setattr(evolve.multiprocessing, "Pool", FakePool)
    archive = evolve.run(terrains="flat", n_init=3, n_batches=1, batch_size=2,
                         workers=1, variants=0, out_dir=str(tmp_path))
    assert archive.cells
    assert all(e.fitness == 1.0 for e in archive.cells.values())
    assert all(rec[2] == 1.0 for rec in archive.lineage.values())


def test_budget_grows_with_joints_up_to_a_cap():
    few, many = mo.template(1), mo.template(2)
    assert evolve.scaled_budget(100, few) == 100 * max(1, few.n_joints / evolve.BUDGET_JOINTS) \
        or few.n_joints > evolve.BUDGET_JOINTS
    big = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    for i in range(1, 17):
        big.nodes[i] = mo.Node(i, "joint", 0 if i <= 8 else i - 8, 0 if i <= 8 else 0)
    assert evolve.scaled_budget(100, big) == round(100 * evolve.BUDGET_SCALE_MAX)
    one = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core"), 1: mo.Node(1, "joint", 0, 0)})
    assert evolve.scaled_budget(100, one) == 100
    assert evolve.scaled_budget(100, many) >= evolve.scaled_budget(100, one)


def test_legs_axis_separates_leg_counts():
    base = [0.5, 0.1, 8.0, 0.0, 5.0]
    assert evolve.cell_of(base + [2.0]) != evolve.cell_of(base + [6.0])

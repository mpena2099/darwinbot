"""Opt-in search options of the second audit round: --face-fix reaching the
run, --mirror-ctrl, --reeval and --dns. All off by default."""
import pickle

import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import evolve
from darwinbot import morphology as mo

BEH = [0.9, 0.3, 8, 0.0, 1.0, 2, 0.0]


@pytest.fixture(autouse=True)
def _reset():
    yield
    mo.set_face_fix(False)


def _tiny(out, **kw):
    return evolve.run(terrains="flat", n_init=3, n_batches=2, batch_size=4,
                      budget_init=4, budget_child=3, workers=1, seed=3,
                      episode_seconds=0.4, variants=0, out_dir=str(out), **kw)


# ---- mirror-ctrl ----

def test_copy_node_genes_copies_block_with_antiphase():
    rng = np.random.default_rng(0)
    c = rng.uniform(0.0, 1.0, co.CTRL_LENGTH)
    out = co.copy_node_genes(c, [(2, 5, False)])
    a, b = co.node_base(2), co.node_base(5)
    n = co.CTRL_PER_NODE
    assert out[b + co.PHASE1] == pytest.approx((c[a + co.PHASE1] + 0.5) % 1.0)
    keep = [i for i in range(n) if i != co.PHASE1]
    assert np.array_equal(out[b:b + n][keep], c[a:a + n][keep])
    assert out[a + co.PHASE1] == c[a + co.PHASE1]       # the source is untouched
    other = np.ones(co.CTRL_LENGTH, bool)
    other[b:b + n] = False
    assert np.array_equal(out[other], c[other])         # nothing else moves
    assert c is not out


def test_mirrored_copy_swaps_steering_sign():
    c = np.full(co.CTRL_LENGTH, 0.3)
    out = co.copy_node_genes(c, [(1, 4, True)])
    b = co.node_base(4)
    assert out[b + co.STEER_GAIN] == pytest.approx(0.7)
    assert out[b + co.STEER_SHIFT] == pytest.approx(0.7)
    assert out[b + co.SPEED_GAIN] == 0.3


def test_copy_node_genes_chains_in_order():
    c = np.zeros(co.CTRL_LENGTH)
    c[co.node_base(1) + co.AMP1] = 0.9
    out = co.copy_node_genes(c, [(1, 2, False), (2, 3, False)])
    assert out[co.node_base(3) + co.AMP1] == 0.9
    assert out[co.node_base(3) + co.PHASE1] == pytest.approx(0.0)   # two half cycles


def test_mutation_logs_copies_only_when_a_limb_is_copied():
    body = mo.encode(mo.template(2))
    seen = 0
    for seed in range(40):
        mo.mutate(body, np.random.default_rng(seed), mirror_p=1.0)
        for src, new, mirrored in mo.copy_log:
            assert src != new
            seen += 1
    assert seen > 0
    mo.mutate(body, np.random.default_rng(1))       # a new mutation starts a new log
    assert all(len(e) == 3 for e in mo.copy_log)


# ---- face-fix ----

def test_face_fix_reaches_search_and_activate(tmp_path):
    a = _tiny(tmp_path, face_fix=True)
    assert a.search["face_fix"] is True
    assert mo._face_fix is True
    mo.set_face_fix(False)
    a.activate()
    assert mo._face_fix is True
    mo.set_face_fix(False)
    evolve._init_worker(face_fix=True)
    assert mo._face_fix is True
    evolve._init_worker()
    assert mo._face_fix is False


def test_defaults_store_the_options_off(tmp_path):
    a = _tiny(tmp_path)
    assert a.search["face_fix"] is False and a.search["mirror_ctrl"] is False
    assert a.search["reeval"] == 0 and a.search["dns"] == 0
    assert a.population == [] and a.reeval == {}


# ---- reeval ----

def test_reeval_mean_counts_the_first_value_once():
    assert evolve.reeval_mean(2.0, []) == 2.0
    assert evolve.reeval_mean(2.0, [1.0]) == pytest.approx(1.5)
    assert evolve.reeval_mean(2.0, [1.0, 0.0]) == pytest.approx(1.0)


def _fill(n=3):
    arch = evolve.Archive(terrains="flat")
    for i in range(n):
        genes = np.full(mo.MORPH_LENGTH, 0.5)
        genes[0] = 0.1 * (i + 1)
        beh = [0.2 + 0.3 * i, 0.3, 8, 0.0, 1.0, 2, 0.0]
        arch.insert(evolve.Elite(genes, np.zeros(co.CTRL_LENGTH), 1.0 + i, [1.0 + i], 0,
                                 np.array(beh)))
    return arch


def test_apply_reeval_averages_and_accumulates():
    arch = _fill(3)
    es = list(arch.cells.values())
    ids = [evolve.elite_id(e.morph, e.ctrl) for e in es]
    first = [e.fitness for e in es]
    res = {ids[0]: (first[0] - 1.0, 0.5), ids[1]: (first[1] - 3.0, 0.5)}
    drop, gap = evolve.apply_reeval(arch, res)
    new = {evolve.elite_id(e.morph, e.ctrl): e.fitness for e in arch.cells.values()}
    assert new[ids[0]] == pytest.approx(first[0] - 0.5)
    assert new[ids[1]] == pytest.approx(first[1] - 1.5)
    assert new[ids[2]] == first[2]                      # not re-scored
    assert drop == pytest.approx(1.0) and gap == pytest.approx(2.0)
    evolve.apply_reeval(arch, {ids[0]: (first[0] - 2.0, 0.5)})   # second re-score
    again = {evolve.elite_id(e.morph, e.ctrl): e.fitness for e in arch.cells.values()}
    assert again[ids[0]] == pytest.approx(first[0] - (1.0 + 2.0) / 3)
    # a worse challenger between the new and old value now wins the cell
    e0 = next(e for e in es if evolve.elite_id(e.morph, e.ctrl) == ids[0])
    challenger = evolve.Elite(e0.morph + 0.0, np.ones(co.CTRL_LENGTH), first[0] - 0.5,
                              [0.0], 1, e0.behaviour)
    assert arch.insert(challenger)


def test_apply_reeval_updates_population_member():
    arch = _fill(2)
    e = next(iter(arch.cells.values()))
    arch.population = [evolve.dns_member(e.morph, e.ctrl, e.fitness, e.behaviour, 0, 1.0)]
    evolve.apply_reeval(arch, {arch.population[0]["id"]: (e.fitness - 2.0, 0.2)})
    assert arch.population[0]["objective"] == pytest.approx(e.fitness - 1.0)
    assert arch.population[0]["progress"] == pytest.approx(0.6)


# ---- dns ----

def _pop(rows):
    """rows: (objective, descriptor x, progress)"""
    out = []
    for i, (obj, x, prog) in enumerate(rows):
        genes = np.full(mo.MORPH_LENGTH, 0.5)
        genes[0] = i / 100
        out.append(evolve.dns_member(genes, np.zeros(co.CTRL_LENGTH), obj,
                                     [x, 0.3, 8, 0.0, 1.0, 2, 0.0], 0, prog))
    return out


def test_dns_fitter_neighbour_removes_the_near_worse_one():
    pop = _pop([(5.0, 0.20, 1.0), (4.0, 0.22, 1.0), (3.0, 0.90, 1.0)])
    kept = evolve.dns_trim(pop, 2)
    assert [m["objective"] for m in kept] == [5.0, 3.0]


def test_dns_isolated_below_floor_member_goes_first():
    pop = _pop([(5.0, 0.20, 1.0), (4.0, 0.22, 1.0), (-1.0, 0.90, 1.0),
                (2.0, 0.60, 0.01)])
    kept = evolve.dns_trim(pop, 2)
    objs = [m["objective"] for m in kept]
    assert -1.0 not in objs and 2.0 not in objs         # both below the floor


def test_dns_isolated_good_member_is_kept():
    pop = _pop([(5.0, 0.20, 1.0), (4.0, 0.21, 1.0), (3.0, 0.22, 1.0),
                (0.5, 0.95, 1.0)])
    kept = evolve.dns_trim(pop, 2)
    objs = [m["objective"] for m in kept]
    assert 0.5 in objs and 5.0 in objs


def test_dns_scores_floor_and_inf():
    s = evolve.dns_scores([3.0, 1.0, 2.0], [[0.0], [0.5], [1.0]], [True, False, True])
    assert np.isinf(s[0]) and s[1] == 0.0
    assert s[2] == pytest.approx(1.0)                   # the one fitter member at 1.0


def test_sample_population_returns_batch_keys():
    pop = _pop([(1.0, 0.1, 1.0), (2.0, 0.5, 1.0), (3.0, 0.9, 1.0)])
    for sel in evolve.PARENT_SELECTIONS:
        out = evolve.sample_population(pop, 5, np.random.default_rng(0), sel)
        assert out["solution"].shape == (5, mo.MORPH_LENGTH)
        assert out["ctrl"].shape == (5, co.CTRL_LENGTH)


# ---- whole run ----

def test_all_options_run_and_resume(tmp_path, capfd):
    kw = dict(dns=20, reeval=2, mirror_ctrl=True, face_fix=True, mirror_p=0.5)
    a = _tiny(tmp_path, **kw)
    out = capfd.readouterr().out
    assert "reeval:" in out
    assert a.population and len(a.population) <= 20
    assert a.reeval
    assert a.search["dns"] == 20 and a.search["reeval"] == 2
    with open(tmp_path / "archive.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert len(loaded.population) == len(a.population)
    b = evolve.run(terrains="flat", n_batches=2, batch_size=4, budget_child=3,
                   workers=1, seed=4, out_dir=str(tmp_path),
                   resume=str(tmp_path / "archive.pkl"), **kw)
    assert b.batch == a.batch + 2 and b.population


def test_default_run_is_bit_identical(tmp_path):
    import hashlib
    a = evolve.run(n_init=3, n_batches=3, batch_size=4, budget_init=6, budget_child=4,
                   workers=1, out_dir=str(tmp_path), seed=5, deterministic=True,
                   mirror_p=0.5)
    h = hashlib.sha1()
    for _, e in sorted(a.cells.items()):
        h.update(np.asarray(e.morph).tobytes())
        h.update(np.asarray(e.ctrl).tobytes())
        h.update(repr(e.fitness).encode())
    # measured on the code before the options existed
    assert h.hexdigest() == "2661c0fe57a6470cece35b274cbc955dba34a4f2"

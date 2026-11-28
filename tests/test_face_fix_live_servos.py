"""face_fix (mutation option) and live-servos (body option), both off by default."""

import math

import mujoco
import numpy as np
import pytest

from darwinbot import body_options, builder, sim
from darwinbot import morphology as mo


@pytest.fixture(autouse=True)
def reset():
    mo.set_face_fix(False)
    body_options.set_active(None)
    yield
    mo.set_face_fix(False)
    body_options.set_active(None)


def _bodies(n, start=0):
    return [mo.decode(mo.random_morphology(np.random.default_rng(seed)))
            for seed in range(start, start + n)]


def _face_bad(morph):
    """Joints and wheels that sit on a core face slot (the case fit_rules repairs)."""
    return [i for i, n in morph.nodes.items()
            if n.parent == 0 and n.slot in mo.FACE_SLOTS and n.kind in ("joint", "wheel")]


def _mutate_all(genes_list, per_body, mirror_p, seed):
    rng = np.random.default_rng(seed)
    return [mo.mutate(g, rng, mirror_p=mirror_p)
            for g in genes_list for _ in range(per_body)]


def _genes(n, start=0):
    return [mo.random_morphology(np.random.default_rng(seed)) for seed in range(start, start + n)]


def _model(morph):
    xml, _ = builder.build_mjcf(morph, ranges={})
    return mujoco.MjModel.from_xml_string(xml)


def test_face_fix_keeps_joints_and_wheels_off_face_slots(monkeypatch):
    bodies = _genes(40)
    before_fit = []
    orig_fit = mo.fit_rules

    def spy(morph):
        before_fit.append(_face_bad(morph))
        return orig_fit(morph)

    monkeypatch.setattr(mo, "fit_rules", spy)

    # switch off: the waste is there (sanity check that the test can see it)
    _mutate_all(bodies, 50, mirror_p=0.3, seed=1)
    assert len(before_fit) >= 2000
    assert any(before_fit)

    # switch on: 2000 mutations, nothing for fit_rules to repair on a face slot
    before_fit.clear()
    mo.set_face_fix(True)
    _mutate_all(bodies, 50, mirror_p=0.3, seed=1)
    assert len(before_fit) >= 2000
    assert not any(before_fit)


def test_face_fix_off_matches_untouched_switch():
    bodies = _genes(20)
    untouched = _mutate_all(bodies, 5, mirror_p=0.3, seed=7)
    mo.set_face_fix(True)
    mo.set_face_fix(False)
    again = _mutate_all(bodies, 5, mirror_p=0.3, seed=7)
    assert len(untouched) == len(again)
    for a, b in zip(untouched, again):
        np.testing.assert_array_equal(a, b)


def test_live_servos_token_parsing():
    assert body_options.parse("").live_servos is False
    assert body_options.BodyOptions().live_servos is False
    opts = body_options.parse("live-servos")
    assert opts.live_servos and not opts.is_default()
    assert opts.to_string() == "live-servos"
    mixed = body_options.parse("no-wheels,live-servos,legs=4..6")
    assert body_options.parse(mixed.to_string()) == mixed
    body_options.set_active("live-servos")
    assert body_options.active().live_servos
    body_options.set_active(None)
    assert not body_options.active().live_servos
    with pytest.raises(ValueError):
        body_options.parse("live-servo")


def test_blocked_servo_rejected_only_with_live_servos(monkeypatch):
    morph = next(m for m in _bodies(60)
                 if any(mo.is_joint(n.kind) for n in m.nodes.values())
                 and sim.is_buildable(_model(m), m))
    blocked = {i: (-builder.BLOCKED_SPAN, builder.BLOCKED_SPAN)
               for i, n in morph.nodes.items() if mo.is_joint(n.kind)}
    monkeypatch.setattr(sim, "joint_ranges", lambda m: blocked)
    model = _model(morph)
    assert sim.is_buildable(model, morph)          # option off: unchanged
    body_options.set_active("live-servos")
    assert not sim.is_buildable(model, morph)      # option on: rejected
    body_options.set_active(None)
    assert sim.is_buildable(model, morph)


def test_live_servos_follows_real_joint_ranges():
    checked = 0
    for morph in _bodies(30):
        model = _model(morph)
        if not sim.is_buildable(model, morph):
            continue
        spans = [hi - lo for i, (lo, hi) in builder.joint_ranges(morph).items()
                 if mo.is_joint(morph.nodes[i].kind)]
        expected = all(s >= math.radians(8.0) for s in spans)
        body_options.set_active("live-servos")
        try:
            got = sim.is_buildable(model, morph)
        finally:
            body_options.set_active(None)
        assert got == expected
        checked += 1
        if checked == 4:
            break
    assert checked == 4

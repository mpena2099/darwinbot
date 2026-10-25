"""
test_sim.py
-----------
MuJoCo episode construction, standing, wheel drive and determinism.
"""

import mujoco
import numpy as np
import pytest

from darwinbot import controller as co
from darwinbot import fitness
from darwinbot import morphology as mo
from darwinbot import randomize
from darwinbot import terrain as te
from darwinbot.sim import Episode, Metrics, cost_of_transport


LEGS = mo.encode(mo.template(2, wheels=False, yaw=True))
WHEELS = mo.encode(mo.template(2, wheels=True, yaw=True, wheel_rad=0.06))


def _ctrl(morph, amp=0.0, wheel=0.5, sigma=0.5):
    """Mid genes everywhere; joints get `amp` on both harmonics, wheels `wheel`."""
    ctrl = np.full(co.CTRL_LENGTH, 0.5)
    for act in morph.layout():
        b = co.node_base(act.slot)
        if act.kind == "wheel":
            ctrl[b + co.OFFSET] = wheel
        else:
            ctrl[b + co.AMP1] = amp
            ctrl[b + co.AMP2] = amp
            ctrl[b + co.SIGMA] = sigma
    return ctrl


@pytest.mark.parametrize("name", list(te.TERRAINS))
def test_every_terrain_compiles(name):
    episode = Episode(mo.decode(LEGS), te.TERRAINS[name](), seconds=1.0)
    assert episode.model.nu == len(episode.layout)


def test_no_yaw_leg_has_one_fewer_actuator():
    with_yaw = Episode(mo.template(2, yaw=True), te.flat(), seconds=1.0)
    no_yaw = Episode(mo.template(2, yaw=False), te.flat(), seconds=1.0)

    assert with_yaw.model.nu - no_yaw.model.nu == 4      # 2 pairs x 2 legs
    assert not any(a.axis == "yaw" for a in no_yaw.layout)
    assert no_yaw.model.nu == len(no_yaw.layout)


def test_spring_is_a_passive_hinge():
    """A spring adds a joint and no actuator, with the flexure stiffness."""
    morph = mo.template(1)
    i = max(morph.nodes) + 1
    morph.nodes[i] = mo.Node(i, "spring", 0, 0, 0.0, axis="pitch",
                             length=0.09, width=0.020, thick=0.006)
    episode = Episode(morph, te.flat(), seconds=1.0)
    model = episode.model
    assert len(episode.layout) == morph.n_servos
    assert morph.n_joints + morph.n_springs == len(episode.layout) + 1
    assert not any(a.slot == i for a in episode.layout)
    assert episode.buildable
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"s{i}")
    assert model.jnt_stiffness[jid] == pytest.approx(
        mo.spring_stiffness(morph.nodes[i]))


def _terminal_body(kind, tail_mass=0.0):
    m = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    m.nodes[1] = mo.Node(1, kind, 0, 0, 0.0, length=0.10, tail_mass=tail_mass)
    m.nodes[2] = mo.Node(2, "joint", 0, 3, 3 * np.pi / 2, "pitch", 0.10)
    return m


@pytest.mark.parametrize("kind", ("tail", "skid", "spike"))
def test_terminal_module_is_a_passive_buildable_body(kind):
    morph = _terminal_body(kind)
    episode = Episode(morph, te.flat(), seconds=1.0)
    assert episode.buildable
    assert len(episode.layout) == morph.n_servos
    assert episode.model.nu == len(episode.layout)
    assert kind not in {a.kind for a in episode.layout}


@pytest.mark.parametrize("kind,friction", (("skid", 0.15), ("spike", 1.6)))
def test_terminal_friction_lives_on_its_own_geom(kind, friction):
    episode = Episode(_terminal_body(kind), te.flat(), seconds=1.0)
    gid = mujoco.mj_name2id(episode.model, mujoco.mjtObj.mjOBJ_GEOM, f"{kind}1")
    assert gid >= 0
    assert episode.model.geom_friction[gid, 0] == pytest.approx(friction)


def test_tail_counterweight_adds_its_mass():
    light = Episode(_terminal_body("tail", 0.02), te.flat(), seconds=1.0)
    heavy = Episode(_terminal_body("tail", 0.08), te.flat(), seconds=1.0)
    assert heavy.mass - light.mass == pytest.approx(0.06)


def test_layout_matches_model_order():
    morph = mo.template(2, wheels=True, yaw=True)
    episode = Episode(morph, te.flat(), seconds=1.0)
    names = [mujoco.mj_id2name(episode.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
             for a in range(episode.model.nu)]
    assert names == [f"{'j' if a.kind == 'joint' else 'w'}{a.slot}"
                     for a in episode.layout]
    assert episode.layout == morph.layout()


def test_mirrored_legs_move_alike():
    """The same signal lifts both thighs and swings both yaw servos forward."""
    morph = mo.template(1, yaw=True)
    episode = Episode(morph, te.flat(), seconds=1.0)
    model, data = episode.model, episode.data
    foot = {i: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"n{i}")
            for i in morph.nodes}
    mujoco.mj_forward(model, data)
    rest = data.xpos.copy()
    for axis, dim in (("yaw", 0), ("pitch", 2)):
        mujoco.mj_resetData(model, data)
        for a, act in enumerate(episode.layout):
            if act.axis == axis:
                data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]] = 0.3
        mujoco.mj_forward(model, data)
        for i, n in morph.nodes.items():
            if n.kind == "joint" and not morph.child_map().get(i):
                assert data.xpos[foot[i]][dim] > rest[foot[i]][dim]


def test_overlapping_legs_are_not_buildable():
    same = mo.template(2, wheels=True, wheel_rad=0.06, body_length=0.15)
    spread = mo.template(2, wheels=True, wheel_rad=0.06)

    assert not Episode(same, te.flat(), seconds=1.0).buildable
    assert Episode(spread, te.flat(), seconds=1.0).buildable


def test_side_modules_hang_from_an_elbow_plate():
    morph = mo.template(2, wheels=True)
    side = [i for i, n in morph.nodes.items()
            if n.parent > 0 and n.slot and morph.nodes[n.parent].kind in ("joint", "link")]
    ep = Episode(morph, te.flat(), seconds=1.0)

    assert side and ep.buildable
    for i in side:
        g = ep.model.geom(f"elbow{i}")
        assert g.bodyid[0] == ep.model.body(f"n{morph.nodes[i].parent}").id


def test_printability_violation_is_not_buildable():
    morph = mo.Morphology(0.4, 0.2, {0: mo.Node(0, "core")})
    morph.nodes[1] = mo.Node(1, "joint", 0, 3, 0.0, "pitch", 0.10,
                             width=0.012, thick=0.003)
    assert mo.violations(morph)
    assert not Episode(morph, te.flat(), seconds=1.0).buildable


def test_unbuildable_body_scores_unstable_without_simulating():
    from darwinbot.inner import optimise_controller
    morph = mo.template(2, wheels=True, wheel_rad=0.06, body_length=0.15)
    res = optimise_controller(morph, [te.flat()], budget=50, seconds=1.0)

    assert res.fitness == fitness.UNSTABLE_SCORE
    assert res.evals == 0


def test_controller_updates_at_50hz():
    morph = mo.decode(LEGS)
    episode = Episode(morph, te.flat(), seconds=1.0)
    controller = episode.reset(_ctrl(morph, amp=1.0))
    assert episode.hold == round(co.CONTROL_DT / episode.model.opt.timestep)

    seen = []
    for _ in range(3 * episode.hold):
        episode.step(controller)
        seen.append(episode.data.ctrl.copy())
    for w in range(3):
        window = seen[w * episode.hold:(w + 1) * episode.hold]
        assert all(np.array_equal(window[0], c) for c in window)
    assert not np.array_equal(seen[0], seen[-1])


def test_zero_amplitude_stands_still():
    morph = mo.decode(LEGS)
    episode = Episode(morph, te.flat(), seconds=1.0)
    metrics = episode.run(_ctrl(morph))

    assert not metrics.fell and not metrics.unstable
    assert abs(metrics.dx) < 0.02


def test_wheels_drive_forward():
    morph = mo.decode(WHEELS)
    episode = Episode(morph, te.flat(), seconds=2.0)
    metrics = episode.run(_ctrl(morph, wheel=0.9))

    assert not metrics.fell and not metrics.unstable
    assert metrics.dx > 0.1


def test_runs_are_deterministic():
    morph = mo.decode(WHEELS)
    ctrl = _ctrl(morph, wheel=0.7)
    first = Episode(morph, te.flat(), seconds=1.0).run(ctrl)
    second = Episode(morph, te.flat(), seconds=1.0).run(ctrl)

    assert first == second


def _metrics(**overrides):
    fields = dict(dx=0.0, dy=0.0, energy=0.0, fell=False, unstable=False,
                  duration=1.0, mass=1.0)
    fields.update(overrides)
    return Metrics(**fields)


def test_unstable_gets_unstable_score():
    assert fitness.score(_metrics(unstable=True, dx=5.0)) == fitness.UNSTABLE_SCORE


def test_fall_applies_penalty():
    upright = fitness.score(_metrics(dx=1.0))
    fallen = fitness.score(_metrics(dx=1.0, fell=True))
    assert upright - fallen == pytest.approx(fitness.FALL_PENALTY)


def test_aggregate_blends_mean_and_min():
    scores = [1.0, 2.0, 6.0]
    expected = 0.7 * np.mean(scores) + 0.3 * min(scores)
    assert fitness.aggregate(scores) == pytest.approx(expected)


def _episode():
    return Episode(mo.decode(LEGS), te.get_terrains("flat")[0], seconds=1.0)


def test_variant_none_is_bit_identical():
    ep = _episode()
    ctrl = np.random.default_rng(3).random(co.CTRL_LENGTH)
    a = ep.run(ctrl)
    nominal = randomize.Variant(ep.model, 0)
    assert ep.run(ctrl, nominal) == a
    assert ep.run(ctrl) == a


def test_latency_delays_first_command(monkeypatch):
    ctrl = np.random.default_rng(3).random(co.CTRL_LENGTH)

    def first_nonzero(latency):
        ep = _episode()
        ep.n_steps = ep.hold * 6
        applied = []
        real = mujoco.mj_step

        def spy(model, data):
            real(model, data)
            if ep._k % ep.hold == 1:          # first step after a control update
                applied.append(np.abs(data.ctrl).max())

        monkeypatch.setattr(mujoco, "mj_step", spy)
        ep.run(ctrl, randomize.Variant(ep.model, latency))
        monkeypatch.undo()
        return next(i for i, a in enumerate(applied) if a > 0)

    assert first_nonzero(2) == first_nonzero(0) + 2


def test_tegotae_changes_the_trajectory():
    morph = mo.decode(LEGS)
    open_loop = Episode(morph, te.flat(), seconds=1.0).run(_ctrl(morph, amp=0.6, sigma=0.5))
    loaded = Episode(morph, te.flat(), seconds=1.0).run(_ctrl(morph, amp=0.6, sigma=1.0))
    again = Episode(morph, te.flat(), seconds=1.0).run(_ctrl(morph, amp=0.6, sigma=1.0))
    assert loaded == again
    assert (loaded.dx, loaded.dy, loaded.energy) != (open_loop.dx, open_loop.dy,
                                                     open_loop.energy)


def test_phases_are_integrated_every_physics_step():
    morph = mo.decode(LEGS)
    ep = Episode(morph, te.flat(), seconds=1.0)
    controller = ep.reset(_ctrl(morph, amp=0.6, sigma=0.5))
    for _ in range(ep.hold + 1):
        ep.step(controller)
    dt = ep.model.opt.timestep
    assert np.all(controller.theta > 0.5 * (ep.hold + 1) * dt * controller.omega)
    assert np.all(controller.theta < 1.5 * (ep.hold + 1) * dt * controller.omega)


def test_behaviour_in_range_and_deterministic():
    for genes, kw in ((LEGS, dict(amp=0.6)), (WHEELS, dict(wheel=0.9))):
        morph = mo.decode(genes)
        episode = Episode(morph, te.flat(), seconds=2.0)
        a = episode.run(_ctrl(morph, **kw))
        b = episode.run(_ctrl(morph, **kw))
        assert 0.0 <= a.duty <= 1.0
        assert 0.0 < a.height <= 1.5
        assert 0.0 <= a.wheel_share <= 1.0
        assert (a.duty, a.height, a.wheel_share) == (b.duty, b.height, b.wheel_share)


def test_rolling_robot_is_named_roll():
    from darwinbot import evolve
    from darwinbot.inner import behaviour_vector
    morph = mo.decode(WHEELS)
    m = Episode(morph, te.flat(), seconds=2.0).run(_ctrl(morph, wheel=0.9))
    assert m.wheel_share > 0.5 and m.dx > 0.1
    assert evolve.type_name(behaviour_vector(morph, m)).endswith("-roll")
    legs = mo.decode(LEGS)
    m = Episode(legs, te.flat(), seconds=2.0).run(_ctrl(legs, amp=0.6))
    assert evolve.type_name(behaviour_vector(legs, m)).endswith(("crawl", "walk", "trot", "belly"))


def test_optimise_returns_nominal_flat_behaviour():
    from darwinbot.inner import optimise_controller
    morph = mo.decode(LEGS)
    res = optimise_controller(morph, [te.flat()], budget=4, seed=1, seconds=1.0, variants=1)
    ref = Episode(morph, te.flat(), seconds=1.0).run(res.ctrl)
    assert np.allclose(res.behaviour, [ref.duty, ref.height, morph.n_servos, ref.wheel_share,
                                       cost_of_transport(ref)])


def test_cost_of_transport_is_clipped_and_charges_standing_still():
    from darwinbot.sim import COT_DX_MIN, COT_MAX, COT_MIN
    moving = Metrics(dx=1.0, dy=0.0, energy=9.81, fell=False, unstable=False,
                     duration=5.0, mass=1.0)
    assert cost_of_transport(moving) == pytest.approx(1.0)
    still = Metrics(dx=-0.2, dy=0.0, energy=9.81 * COT_DX_MIN, fell=False,
                    unstable=False, duration=5.0, mass=1.0)
    assert cost_of_transport(still) == pytest.approx(1.0)
    idle = Metrics(0.0, 0.0, 0.0, False, False, 5.0, 1.0)
    assert cost_of_transport(idle) == COT_MIN
    assert cost_of_transport(Metrics(0.0, 0.0, 1e6, False, False, 5.0, 1.0)) == COT_MAX


def test_skid_and_spike_friction_beats_the_terrain():
    morph = mo.template(2)
    for kind in ("skid", "spike"):
        i = max(morph.nodes) + 1
        morph.nodes[i] = mo.Node(i, kind, 0, 0 if kind == "skid" else 1, 0.0)
    ep = Episode(morph, te.flat(), seconds=1.0)
    floor = max(ep.model.geom_priority[g] for g in range(ep.model.ngeom)
                if ep.model.geom_contype[g] != 2)
    for name in ("skid", "spike"):
        g = next(g for g in range(ep.model.ngeom) if (ep.model.geom(g).name or "").startswith(name))
        assert ep.model.geom_priority[g] > floor


def test_caster_is_a_free_wheel_without_actuator():
    morph = mo.template(2)
    i = max(morph.nodes) + 1
    morph.nodes[i] = mo.Node(i, "caster", 0, 1, 0.0, wheel_rad=0.025)
    assert morph.n_servos == mo.template(2).n_servos
    ep = Episode(morph, te.flat(), seconds=1.0)
    assert ep.buildable
    hinges = sum(ep.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    assert hinges == ep.model.nu + 1
    assert mo.decode(mo.encode(morph)).nodes[i].kind == "caster"

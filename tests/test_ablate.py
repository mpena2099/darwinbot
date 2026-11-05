import mujoco
import numpy as np
import pytest

from darwinbot import ablate, builder, evolve, morphology as mo
from darwinbot import terrain as te
from darwinbot.sim import Episode, is_buildable


@pytest.fixture(autouse=True)
def reset():
    ablate.set_active(())
    yield
    ablate.set_active(())


def _joint_ranges_of(morph):
    xml, _ = builder.build_mjcf(morph)
    m = mujoco.MjModel.from_xml_string(xml)
    return m, m.jnt_range[m.jnt_limited.astype(bool)]


def _default_ranges(morph):
    xml, _ = builder.build_mjcf(morph, ranges={})
    m = mujoco.MjModel.from_xml_string(xml)
    return m.jnt_range[m.jnt_limited.astype(bool)]


def _bodies(n=40):
    rng = np.random.default_rng(3)
    return [mo.decode(mo.random_morphology(rng)) for _ in range(n)]


def test_names_validated_and_default_empty():
    assert ablate.normalise(["friction", "ranges"]) == ("ranges", "friction")
    with pytest.raises(ValueError):
        ablate.normalise(["nonsense"])
    assert not any(ablate.active(n) for n in ablate.NAMES)
    assert evolve.Archive(terrains="flat").ablate == ()


def test_old_archive_loads_with_empty_ablate():
    a = evolve.Archive(terrains="flat")
    state = dict(a.__dict__)
    del state["ablate"]
    b = evolve.Archive.__new__(evolve.Archive)
    b.__setstate__(state)
    assert b.ablate == ()


def test_ranges_ablation_uses_default_joint_range():
    shrunk = None
    for body in _bodies():
        _, got = _joint_ranges_of(body)
        if got.shape != _default_ranges(body).shape or not np.allclose(got, _default_ranges(body)):
            shrunk = body
            break
    assert shrunk is not None, "no random body had a shortened joint range"
    ablate.set_active(["ranges"])
    _, got = _joint_ranges_of(shrunk)
    assert np.allclose(got, _default_ranges(shrunk))


def test_friction_ablation_makes_plastic_grip_like_tpu():
    xml, _ = builder.build_mjcf(mo.template(2))
    m = mujoco.MjModel.from_xml_string(xml)
    robot = m.geom_contype == 2
    assert (m.geom_friction[robot & (m.geom_priority == 2), 0] == builder.PLASTIC_FACTOR).all()
    ablate.set_active(["friction"])
    xml, _ = builder.build_mjcf(mo.template(2), terrain_friction=0.8)
    m = mujoco.MjModel.from_xml_string(xml)
    plastic = robot & (m.geom_priority == 2)
    assert plastic.any()
    assert np.allclose(m.geom_friction[plastic, 0], 0.8)


def test_assembly_ablation_allows_printed_part_overlaps_only():
    flat = te.flat()
    found = None
    rng = np.random.default_rng(5)
    raw = [mo.decode(rng.random(mo.MORPH_LENGTH)) for _ in range(150)]   # not fitted
    for body in raw:
        if mo.violations(body):
            continue
        ep = Episode(body, flat, seconds=0.1)
        if not ep.buildable:
            ablate.set_active(["assembly"])
            ok = is_buildable(ep.model, body)
            ablate.set_active(())
            if ok:
                found = body
                break
    assert found is not None, "no random body was rescued by the assembly ablation"
    assert not Episode(found, flat, seconds=0.1).buildable
    # a part sunk into its parent (wheel inside the torso) is still a servo/mount matter
    assert Episode(mo.template(2), flat, seconds=0.1).buildable


def test_pool_worker_receives_the_ablations():
    evolve._init_worker(("friction", "ranges"))
    assert ablate.active("friction") and ablate.active("ranges")
    assert not ablate.active("assembly")

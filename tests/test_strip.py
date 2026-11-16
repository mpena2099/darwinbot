import importlib.util
import pathlib

import mujoco
import pytest

from darwinbot import terrain as te

_spec = importlib.util.spec_from_file_location(
    "strip_test", pathlib.Path(__file__).parent.parent / "experiments" / "strip_test.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def test_strip_is_where_asked_with_its_friction():
    t = te.get_terrains("strip:at=0.8:length=0.4:mu=0.15")[0]
    xml = (f'<mujoco><asset><material name="grid"/></asset><worldbody>{t.world}'
           '</worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    g = m.geom("strip")
    assert g.pos[0] == pytest.approx(1.0)
    assert g.size[0] == pytest.approx(0.2)
    assert g.friction[0] == pytest.approx(0.15)
    assert g.pos[2] + g.size[2] <= 0.0011       # top at most 1 mm above the plane


def test_strip_defaults_and_not_in_presets():
    t = te.strip()
    assert 'friction="0.2 ' in t.world
    assert all("strip" not in names for names in te.PRESETS.values())
    assert "strip" in te.EVAL_ONLY


def test_parameter_parsing():
    assert te.parse_spec("strip:at=0.3:mu=0.1") == [("strip", {"at": 0.3, "mu": 0.1})]
    with pytest.raises(ValueError):
        te.parse_spec("strip:width=1")


def test_verdict_pass_fail_and_not_reached():
    ok = st.Row(0.3, True, 21, 1.0, 0.9)
    bad = st.Row(0.5, True, 21, 0.9, 0.5)
    far = st.Row(1.2, False)
    assert st.verdict([ok, far]) == (True, [])
    assert st.verdict([ok, bad, far]) == (False, [0.5])
    assert st.verdict([far])[0] is False
    assert st.verdict([bad], min_pass=0.5) == (True, [])


def test_passes_needs_upright_and_a_body_past_the_edge():
    assert st.passes(1.5, False, 1.0, 0.4)
    assert not st.passes(1.3, False, 1.0, 0.4)
    assert not st.passes(1.5, True, 1.0, 0.4)

"""Organic check: hollowed plates map onto the builder's masses and re-simulate."""

import mujoco
import numpy as np

from darwinbot import builder
from darwinbot import controller as co
from darwinbot import morphology as mo
from darwinbot.export import link_args, link_name, organic_check, organic_factors

FLAT = "flat"


def _walker():
    """The template body plus one plain link on a free core slot, ending in a
    skid: no servo, foot, fork or fin, so its plate is one the export hollows."""
    m = mo.template(2)
    used = {n.slot for n in m.nodes.values() if n.parent == 0}
    slot = next(s for s in range(8) if s not in used)
    link = max(m.nodes) + 1
    m.nodes[link] = mo.Node(link, "link", 0, slot, 0, length=0.10, width=0.02)
    m.nodes[link + 1] = mo.Node(link + 1, "skid", link, 0)
    return m, link


def _part_name(morph, i):
    length, width, thick, profile, bend = link_args(morph, i)[:5]
    return f"{link_name(profile, bend, length, width, thick)}.stl"


def _mass(morph, factors=None):
    with builder.mass_factors(factors or {}):
        xml = builder.build_mjcf(morph)[0]
    return mujoco.MjModel.from_xml_string(xml).body_mass.sum()


def test_organic_factors_maps_a_hollowed_link_to_its_node():
    m, link = _walker()
    assert organic_factors(m, {_part_name(m, link): 0.5}) == {("link", link): 0.5}
    assert organic_factors(m, {"link_1x1x1.stl": 0.5}) == {}


def test_hollow_link_is_lighter_and_the_builder_is_back_to_solid_after():
    m, link = _walker()
    solid = _mass(m)
    assert _mass(m, {("link", link): 0.5}) < solid
    assert _mass(m) == solid


def test_organic_check_returns_two_scores_and_unit_ratios_change_nothing():
    m, link = _walker()
    ctrl = np.random.default_rng(0).uniform(0, 1, co.CTRL_LENGTH)
    solid, hollow = organic_check(m, ctrl, {_part_name(m, link): 1.0}, FLAT, 1.0)
    assert isinstance(solid, float) and isinstance(hollow, float)
    assert solid == hollow

"""Tests for the URDF export (darwinbot/urdf.py)."""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from darwinbot import builder
from darwinbot import morphology as mo
from darwinbot import robot_file, urdf
from darwinbot.terrain import get_terrains

CAR = os.path.join(os.path.dirname(__file__), "..", "examples", "robots", "two-wheel-car.yaml")
WHEEL_SEED = 2                      # a joint and a wheel, no other passive module
PLAIN_SEEDS = (5, 14)               # core and joints only, no wheel


def _decoded(seed):
    return mo.decode(mo.random_morphology(np.random.default_rng(seed)))


def _check(morph, path, payload=0.0):
    urdf.write_urdf(morph, str(path), payload)
    flat = get_terrains("flat")[0]
    xml, layout = builder.build_mjcf(morph, flat.assets, flat.world, payload=payload)
    model = mujoco.MjModel.from_xml_string(xml)
    root = ET.parse(path).getroot()

    links = [link.get("name") for link in root.findall("link")]
    assert len(links) == len(set(links)) == model.nbody - 1
    joints = root.findall("joint")
    names = [j.get("name") for j in joints]
    assert len(names) == len(set(names))
    parents = {j.find("parent").get("link") for j in joints}
    children = {j.find("child").get("link") for j in joints}
    assert parents | children <= set(links)
    assert set(links) - children == {"torso"}

    masses = [float(link.find("inertial/mass").get("value")) for link in root.findall("link")]
    assert all(m > 0 for m in masses)
    assert sum(masses) == pytest.approx(float(model.body_mass.sum()), rel=0.01)

    hinges = {j.get("name"): j for j in joints if j.get("type") != "fixed"}
    mujoco_hinges = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                     for j in range(1, model.njnt)}
    assert set(hinges) == mujoco_hinges
    actuated = [n for n in hinges if n[0] in "jw"]
    assert len(actuated) == len(layout)
    for name, joint in hinges.items():
        if name[0] == "w":
            assert joint.get("type") == "continuous"
        if name[0] == "j":
            assert joint.get("type") == "revolute"
            lower, upper = (float(joint.find("limit").get(k)) for k in ("lower", "upper"))
            assert lower < upper
    return root


def test_urdf_random_with_wheel(tmp_path):
    _check(_decoded(WHEEL_SEED), tmp_path / "wheel.urdf")


@pytest.mark.parametrize("seed", PLAIN_SEEDS)
def test_urdf_random_without_wheel(tmp_path, seed):
    _check(_decoded(seed), tmp_path / f"plain_{seed}.urdf")


def test_urdf_two_wheel_car(tmp_path):
    morph, _ = robot_file.load(CAR)
    root = _check(morph, tmp_path / "car.urdf")
    assert any(j.get("type") == "continuous" for j in root.findall("joint"))

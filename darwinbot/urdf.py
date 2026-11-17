"""
urdf.py
-------
A plain URDF file of a robot, for ROS tools (rviz, Gazebo) and other
simulators. It is read from the MuJoCo model that robot.xml describes, so the
links, joints and masses are the simulated robot's. The torso is the root link.

Collision and visual are simple shapes (box, cylinder, sphere; a capsule is a
cylinder), not the printed STLs. Mesh geoms (the torso plates) are skipped, and
so are the massless geoms that only exist for the assembly check (cradles,
bosses, U forks, horn discs): they have density 0 in the MJCF.
"""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from darwinbot import builder
from darwinbot.controller import SERVO_SPEED, SERVO_TORQUE
from darwinbot.terrain import get_terrains

ROOT = "torso"
MIN_MASS = 1e-6         # kg: a body with no geoms of its own (a wheel's hinge frame)
MIN_INERTIA = 1e-12     # kg m^2: keeps the inertia valid for such a body
SHAPES = {int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_SPHERE),
          int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)}


def _num(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _rpy(quat) -> np.ndarray:
    """URDF roll, pitch, yaw (R = Rz(yaw) Ry(pitch) Rx(roll)) of a MuJoCo quaternion."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=float))
    r = rot.reshape(3, 3)
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    if abs(r[2, 0]) < 1 - 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:                               # gimbal lock: roll is taken as zero
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return np.array([roll, pitch, yaw])


def _origin(pos, quat) -> ET.Element:
    return ET.Element("origin", xyz=_num(pos), rpy=_num(_rpy(quat)))


def _under(model, body: int, root: int) -> bool:
    while body > 0:
        if body == root:
            return True
        body = model.body_parentid[body]
    return False


def _unique(name: str, used: set) -> str:
    out, k = name, 1
    while out in used:
        k += 1
        out = f"{name}_{k}"
    used.add(out)
    return out


def _mass_bearing(xml: str, model) -> list:
    """Per MuJoCo geom id, whether the geom has mass. A geom belongs to the
    nearest enclosing body element (a joint's servo sits in its parent's frame),
    and MuJoCo keeps the geoms of each body in document order. Names are checked
    so a mismatch fails loudly."""
    per_body = {}

    def walk(el, body: str) -> None:
        for child in el:
            if child.tag == "geom":
                massless = child.get("density") == "0" and "mass" not in child.attrib
                per_body.setdefault(body, []).append((child.get("name"), not massless))
            else:
                walk(child, child.get("name") if child.tag == "body" else body)

    walk(ET.fromstring(xml).find("worldbody"), "worldbody")
    seen, out = {}, []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        k = seen.get(b, 0)
        seen[b] = k + 1
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) if b else "worldbody"
        name, mass = per_body[body][k]
        assert name == mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        out.append(mass)
    return out


def _geometry(model, g: int) -> ET.Element:
    kind, size = int(model.geom_type[g]), model.geom_size[g]
    geo = ET.Element("geometry")
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):          # MuJoCo gives half sizes
        ET.SubElement(geo, "box", size=_num(2 * size[:3]))
    elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        ET.SubElement(geo, "sphere", radius=_num(size[:1]))
    else:                                               # cylinder, capsule
        ET.SubElement(geo, "cylinder", radius=_num(size[:1]), length=_num(2 * size[1:2]))
    return geo


def _mesh_box(model, g: int):
    """A mesh geom as the box that bounds it: MuJoCo keeps the bounding box
    (centre, half sizes) in the geom frame."""
    centre, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    pos = model.geom_pos[g] + rot.reshape(3, 3) @ centre
    geo = ET.Element("geometry")
    ET.SubElement(geo, "box", size=_num(2 * half))
    return _origin(pos, model.geom_quat[g]), geo


def _inertial(model, body: int) -> ET.Element:
    el = ET.Element("inertial")
    el.append(_origin(model.body_ipos[body], model.body_iquat[body]))
    ET.SubElement(el, "mass", value=_num([max(model.body_mass[body], MIN_MASS)]))
    ixx, iyy, izz = np.maximum(model.body_inertia[body], MIN_INERTIA)
    ET.SubElement(el, "inertia", ixx=_num([ixx]), iyy=_num([iyy]), izz=_num([izz]),
                  ixy="0", ixz="0", iyz="0")
    return el


def write_urdf(morph, path, payload: float = 0.0) -> None:
    """Write the robot of morph (a decoded Morphology) as a URDF file at path,
    built from the same MJCF as robot.xml on the flat terrain."""
    flat = get_terrains("flat")[0]
    xml, _ = builder.build_mjcf(morph, flat.assets, flat.world, payload=payload)
    model = mujoco.MjModel.from_xml_string(xml)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT)
    robot = [b for b in range(1, model.nbody) if _under(model, b, root_id)]
    has_mass = _mass_bearing(xml, model)

    links, joints = {}, set()
    link_names = set()
    for b in robot:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
        links[b] = _unique(name, link_names)

    urdf = ET.Element("robot", name="darwinbot")
    for b in robot:
        link = ET.SubElement(urdf, "link", name=links[b])
        link.append(_inertial(model, b))
        for g in np.flatnonzero(model.geom_bodyid == b):
            if not has_mass[g]:
                continue     # massless helpers are not drawn
            if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                origin, geo = _mesh_box(model, g)    # the torso plates: their bounding box
            elif int(model.geom_type[g]) in SHAPES:
                origin, geo = _origin(model.geom_pos[g], model.geom_quat[g]), _geometry(model, g)
            else:
                continue
            for tag in ("visual", "collision"):
                part = ET.SubElement(link, tag)
                part.append(origin)
                part.append(geo)

    for b in robot:
        if b == root_id:
            continue                    # the torso is the root: its free joint is dropped
        parent, child = links[model.body_parentid[b]], links[b]
        pos, quat = model.body_pos[b], model.body_quat[b]
        if model.body_jntnum[b] == 0:
            name = _unique(f"fixed_{child}", joints)
            joint = ET.SubElement(urdf, "joint", name=name, type="fixed")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            joint.append(_origin(pos, quat))
            continue
        j = model.body_jntadr[b]
        # a hinge's pivot is the body origin: the MJCF gives no joint offset
        assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        assert np.allclose(model.jnt_pos[j], 0.0)
        name = _unique(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}",
                       joints)
        limited = bool(model.jnt_limited[j])
        joint = ET.SubElement(urdf, "joint", name=name,
                              type="revolute" if limited else "continuous")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        joint.append(_origin(pos, quat))
        ET.SubElement(joint, "axis", xyz=_num(model.jnt_axis[j]))
        # URDF wants an effort and a velocity bound: the servo's, for every joint
        limit = {"effort": _num([SERVO_TORQUE]), "velocity": _num([SERVO_SPEED])}
        if limited:
            lower, upper = model.jnt_range[j]
            limit = {"lower": _num([lower]), "upper": _num([upper]), **limit}
        ET.SubElement(joint, "limit", **limit)

    ET.indent(urdf, space="  ")
    ET.ElementTree(urdf).write(path, encoding="utf-8", xml_declaration=True)

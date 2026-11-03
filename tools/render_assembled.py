"""
Render a robot from the archive built from its printed parts: the STLs that
export.py writes (torso plate, links, elbow plates, servo brackets, feet,
wheels) placed where the simulation puts the modules, with the servo bodies
and the electronics as plain boxes. For judging how the real robot will look,
not for physics.

    python tools/render_assembled.py --archive a.pkl --type 7servos-mid-trot
    python tools/render_assembled.py --archive a.pkl --type <type> --t 1.5 --out robot.png

Writes one image with four views: the standing pose and the pose `--t`
seconds into a straight walk, each from the side and from above at an angle.
Passive modules other than feet and wheels (springs, tails, skids, spikes,
casters) and spine segments keep their simulation shapes.
"""

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from assembly import placements                             # noqa: E402
from replay import load                                     # noqa: E402

from darwinbot import builder, parts, morphology as mo   # noqa: E402
from darwinbot.sim import Episode                          # noqa: E402
from darwinbot.terrain import get_terrains                 # noqa: E402

COLOUR = {"pla": "0.93 0.55 0.15 1",     # orange PLA, stands out from the floor
          "spring": builder.SPRING_RGBA,
          "tpu": "0.12 0.12 0.12 1"}
SERVO = "0.10 0.10 0.10 1"
BOARD = "0.10 0.45 0.20 1"
BATTERY = "0.20 0.25 0.60 1"
WALL = 0.0025                    # bracket wall, as builder._servo_dress


def _mat_quat(rot) -> str:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=float).reshape(9))
    return " ".join(f"{v:.6f}" for v in q)


class Meshes:
    """STL files written once per distinct part, as MuJoCo mesh assets."""

    def __init__(self, folder):
        self.folder, self.assets, self.done = folder, [], set()

    def get(self, name, make):
        if name not in self.done:
            parts.export_stl(make(), os.path.join(self.folder, name + ".stl"))
            self.assets.append(ET.Element("mesh", name=name, file=name + ".stl",
                                          scale="0.001 0.001 0.001"))
            self.done.add(name)
        return name


def _mesh_geom(mesh, pos, rot, rgba):
    return ET.Element("geom", type="mesh", mesh=mesh,
                      pos=" ".join(f"{v:.6f}" for v in pos), quat=_mat_quat(rot),
                      rgba=rgba, contype="0", conaffinity="0", density="0")


def _hide(geom):
    geom.set("rgba", "0 0 0 0")


def _is_plate(geom):
    """Link pieces and elbow plates: the PLA (or TPU spring) boxes."""
    return geom.get("type") == "box" and geom.get("rgba") in (
        builder.LINK_RGBA, builder.SPRING_RGBA) and geom.get("group") != "1"


def assembled_xml(morph, folder) -> str:
    xml, _ = builder.build_mjcf(morph)
    root = ET.fromstring(xml)
    meshes = Meshes(folder)
    bodies = {b.get("name"): b for b in root.iter("body")}
    cm = morph.child_map()
    sz = np.array(mo.SERVO_BOX)[2] / 2

    # torso: hide the simulated prism, put the plate under the edge brackets
    torso = bodies["torso"]
    for g in list(torso.findall("geom")):
        _hide(g)
    floor_z = -sz - WALL                       # bracket floor = top of the plate
    ex, ey = mo.ELECTRONICS_BOX
    torso.append(ET.Element("geom", type="box", pos=f"0 0 {floor_z + 0.004:.4f}",
                            size=f"{0.029:.4f} {0.0115:.4f} 0.004", rgba=BOARD,
                            contype="0", conaffinity="0", density="0"))
    torso.append(ET.Element("geom", type="box", pos=f"{-ex / 4:.4f} 0 {floor_z + 0.023:.4f}",
                            size=f"{ex / 4:.4f} {ey / 2.5:.4f} 0.011", rgba=BATTERY,
                            contype="0", conaffinity="0", density="0"))

    # sim shapes the printed parts replace: bracket visual, plate boxes, caps
    for i, node in morph.nodes.items():
        body = bodies.get(f"n{i}")
        if body is None:
            continue
        geoms = body.findall("geom")
        if node.kind in ("joint", "wheel", "segment"):
            for g in geoms:              # servo body darker; bracket from the STL
                if g.get("rgba") == builder.BRACKET_RGBA:
                    _hide(g)
                elif g.get("type") == "box" and g.get("mass") is not None:
                    g.set("rgba", SERVO)
        if node.kind in ("joint", "link", "spring"):
            for g in geoms:
                if _is_plate(g):
                    _hide(g)
            if i not in cm and node.kind != "spring":       # free end: TPU cap
                for g in body.findall("geom"):
                    if g.get("type") == "sphere":
                        _hide(g)
        elif node.kind in ("foot", "wheel"):
            for g in (body.find("body") if node.kind == "wheel" else body).findall("geom"):
                _hide(g)

    for p in placements(morph):
        m = meshes.get(p.name, p.make)
        bodies[p.body].append(_mesh_geom(m, np.array(p.pos) / 1000, p.rot, COLOUR[p.kind]))
    for i, node in morph.nodes.items():         # wheel tyre, around the plate
        if node.kind == "wheel":
            bodies[f"w{i}"].append(ET.Element(
                "geom", type="cylinder", fromto="0 -0.002 0 0 0.002 0",
                size=f"{node.wheel_rad + 0.0015:.4f}", rgba=COLOUR["tpu"],
                contype="0", conaffinity="0", density="0"))

    asset = root.find("asset")
    for a in meshes.assets:
        asset.append(a)
    root.find("compiler").set("meshdir", folder)
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.95 0.97 1.0", rgb2="0.65 0.72 0.82", width="256", height="256")
    world = root.find("worldbody")
    world.insert(0, ET.Element("geom", type="plane", size="3 3 0.1", material="grid",
                               contype="0", conaffinity="0"))
    world.insert(0, ET.Element("light", pos="1 -1 3", dir="-0.3 0.3 -1",
                               diffuse="0.5 0.5 0.5", castshadow="true"))
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.4 0.4 0.4")
    return ET.tostring(root, encoding="unicode")


def walk_pose(morph, ctrl, seconds, payload=0.0):
    """qpos after `seconds` of a straight walk on flat ground."""
    ep = Episode(morph, get_terrains("flat")[0], payload=payload)
    cpg = ep.reset(ctrl, command=0.0)
    for _ in range(int(seconds / ep.model.opt.timestep)):
        ep.step(cpg)
    return ep.data.qpos.copy()


def render(model, qpos, size=600):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    com = data.subtree_com[1].copy()
    reach = max(0.25, float(np.max(np.abs(data.geom_xpos[:, :2] - com[:2])
                                   * (model.geom_rgba[:, 3] > 0)[:, None])) * 3.4)
    out = []
    with mujoco.Renderer(model, height=size, width=size) as r:
        for az, el in ((90, -12), (135, -35)):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = com
            cam.distance, cam.azimuth, cam.elevation = reach, az, el
            opt = mujoco.MjvOption()
            r.update_scene(data, camera=cam, scene_option=opt)
            out.append(r.render())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True)
    p.add_argument("--type", required=True, dest="cell")
    p.add_argument("--t", type=float, default=1.5, help="walk time of the second pose, s")
    p.add_argument("--out", default=None, help="image path (default: <type>-assembled.png)")
    a = p.parse_args()
    archive, elite = load(a.archive, a.cell)
    morph = mo.decode(elite.morph)
    with tempfile.TemporaryDirectory() as folder:
        model = mujoco.MjModel.from_xml_string(assembled_xml(morph, folder))
    stand = np.array(model.qpos0)
    walk = walk_pose(morph, elite.ctrl, a.t, archive.payload)
    imgs = render(model, stand) + render(model, walk)
    import imageio
    top = np.concatenate(imgs[:2], axis=1)
    bottom = np.concatenate(imgs[2:], axis=1)
    out = a.out or f"{a.cell}-assembled.png"
    imageio.imwrite(out, np.concatenate([top, bottom], axis=0))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

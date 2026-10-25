"""
tools/make_site.py
------------------
Build the static gallery (docs/site) from a MAP-Elites archive: an interactive
map of the robots and a 3D replay of each one in the browser (three.js).

  nice -n 19 python tools/make_site.py
  nice -n 19 python tools/make_site.py --max 12 --archive checkpoints/archive.pkl
  python -m http.server -d docs/site

Every buildable elite runs its nominal flat episode (4 s, its own controller);
body poses are recorded at 25 Hz. Output:

  index.html          copied from tools/site/
  data/index.json     one entry per robot (name, fitness, descriptors, file)
  data/<name>.json    robot geoms (MuJoCo types, half sizes, local pose, rgba)
                      and frames: per body [x y z qw qx qy qz], flattened

MuJoCo is Z-up and quaternions are (w, x, y, z); the page converts.
"""

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import mujoco   # noqa: E402
import numpy as np   # noqa: E402

from replay import open_archive   # noqa: E402

from darwinbot.fitness import UNSTABLE_SCORE   # noqa: E402
from darwinbot.sim import Episode   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402
from darwinbot import morphology as morpho   # noqa: E402

SECONDS = 4.0
FPS = 25
GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
}


def r4(a):
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def geometry(model):
    """Robot geoms of a compiled model (the floor and terrain are skipped)."""
    out = []
    for g in range(model.ngeom):
        # robot geoms: physical (contype 2) and look-only parts (render group 1)
        if model.geom_contype[g] != 2 and model.geom_group[g] != 1:
            continue
        kind = GEOM_TYPES.get(model.geom_type[g])
        if kind is None:
            continue
        out.append({"body": int(model.geom_bodyid[g]), "type": kind,
                    "size": r4(model.geom_size[g]), "pos": r4(model.geom_pos[g]),
                    "quat": r4(model.geom_quat[g]), "rgba": r4(model.geom_rgba[g])})
    return out


def record(ep, ctrl):
    """Poses of every body at FPS during the episode: list of flat lists."""
    m, d = ep.model, ep.data
    controller = ep.reset(ctrl)
    every = max(1, round(1.0 / FPS / m.opt.timestep))
    frames = []
    for n in range(ep.n_steps):
        if n % every == 0:
            frames.append(r4(np.concatenate([d.xpos, d.xquat], axis=1).reshape(-1)))
        ep.step(controller)
        if not np.isfinite(d.qacc).all():
            break
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", default=os.path.join(ROOT, "checkpoints", "archive.pkl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "site"))
    ap.add_argument("--max", type=int, default=None, help="keep the best N robots")
    args = ap.parse_args()

    archive = open_archive(args.archive)
    names = archive.names()
    items = sorted(((names[c], e) for c, e in archive.cells.items()
                    if e.fitness > UNSTABLE_SCORE), key=lambda kv: -kv[1].fitness)
    if args.max:
        items = items[:args.max]
    if not items:
        raise SystemExit("no buildable robot in the archive yet")

    flat = get_terrains("flat")[0]
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    index = []
    for name, elite in items:
        ep = Episode(morpho.decode(elite.morph), flat, SECONDS)
        if not ep.buildable:
            print(f"skip {name}: not buildable")
            continue
        duty, height, servos, wheels, cot = (float(v) for v in elite.behaviour[:5])
        fname = f"{name}.json"
        data = {"name": name, "fps": FPS, "nbody": int(ep.model.nbody),
                "geoms": geometry(ep.model), "frames": record(ep, elite.ctrl)}
        with open(os.path.join(args.out, "data", fname), "w") as f:
            json.dump(data, f, separators=(",", ":"))
        index.append({"name": name, "fitness": round(float(elite.fitness), 4),
                      "duty": round(duty, 4), "height": round(height, 4),
                      "servos": int(round(servos)), "wheels": round(wheels, 4),
                      "cot": round(cot, 4), "file": fname})
        print(f"{name}: {len(data['frames'])} frames, {len(data['geoms'])} geoms")
    with open(os.path.join(args.out, "data", "index.json"), "w") as f:
        json.dump(index, f, separators=(",", ":"))
    shutil.copy(os.path.join(ROOT, "tools", "site", "index.html"),
                os.path.join(args.out, "index.html"))
    print(f"{len(index)} robots -> {args.out}")


if __name__ == "__main__":
    main()

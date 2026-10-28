"""
Render every training environment on one image, with the same camera and the
same robot standing at the spawn for scale.

    python tools/terrain_sheet.py --archive checkpoints/archive.pkl
    python tools/terrain_sheet.py --archive <archive.pkl> --type 8servos-low-roll --out terrains.png

The robot is the archive's best elite unless --type picks another. Random
environments (rough, rubble) show the fixed layout that replay and validate
use. Environments of the `terrain` preset, the default training set, get a
highlighted title.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import matplotlib   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import mujoco   # noqa: E402

from replay import open_archive   # noqa: E402

from darwinbot import morphology as morpho   # noqa: E402
from darwinbot.sim import Episode   # noqa: E402
from darwinbot.terrain import PRESETS, TERRAINS, get_terrains   # noqa: E402

CAPTIONS = {               # what a still image cannot show
    "flat": "flat floor", "rough": "bumps up to 4 cm", "steps": "six 3 cm steps",
    "rubble": "rocks 3 to 7 cm", "ledges": "four 6 cm steps", "step": "one 3 cm step",
    "slope": "8 degree ramp", "steep": "15 degree ramp", "downhill": "10 degrees down",
    "slippery": "friction 0.2, like ice", "soft": "yielding, like sand",
}
HIGHLIGHT = "#1f6fbf"


def render(morph, ctrl, name, w=480, h=300):
    ep = Episode(morph, get_terrains(name)[0])
    cpg = ep.reset(ctrl)
    while ep.data.time < 0.5:            # let the robot land on the ground
        ep.step(cpg)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.75, 0.0, 0.05)    # spawn at x = 0, robots walk towards +x
    cam.distance, cam.azimuth, cam.elevation = 2.3, 35, -22
    r = mujoco.Renderer(ep.model, height=h, width=w)
    r.update_scene(ep.data, camera=cam)
    img = r.render().copy()
    r.close()
    return img


def main():
    p = argparse.ArgumentParser(description="darwinbot: one image with every environment")
    p.add_argument("--archive", required=True, help="archive with the robot shown for scale")
    p.add_argument("--type", default=None, help="robot type to show (default: the best)")
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "img", "terrains.png"))
    a = p.parse_args()

    archive = open_archive(a.archive)
    names = archive.names()
    cells = sorted(archive.cells.items(), key=lambda kv: -kv[1].fitness)
    cell, elite = next(((c, e) for c, e in cells if a.type in (None, names[c])), (None, None))
    if elite is None:
        raise SystemExit(f"no robot of type {a.type} in {a.archive}")
    morph = morpho.decode(elite.morph)

    names_ = [n for n in CAPTIONS if n in TERRAINS]
    default = set(PRESETS["terrain"])
    cols = 4
    rows = -(-len(names_) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.45), dpi=100,
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, names_):
        ax.imshow(render(morph, elite.ctrl, name))
        main_set = name in default
        ax.set_title(f"{name}: {CAPTIONS[name]}" + ("  (default set)" if main_set else ""), fontsize=9.5,
                     color=HIGHLIGHT if main_set else "black",
                     fontweight="bold" if main_set else "normal")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005,
                        wspace=0.02, hspace=0.14)
    fig.savefig(a.out, dpi=100)
    plt.close(fig)
    print(f"wrote {a.out} ({len(names_)} environments, robot {names[cell]})")


if __name__ == "__main__":
    main()

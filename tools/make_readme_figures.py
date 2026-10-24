"""
tools/make_readme_figures.py
----------------------------
Regenerate the README figures in docs/img/ from a MAP-Elites archive.

  robots.png        grid of elites with different body plans, mid-gait
  gait.gif          best wheel-less elite walking, camera following the torso
  archive_map.png   archive heatmap (plot_archive.heatmap)
  progress.png      fitness and filled cells over time (plot_archive.history)
  parts.png         printable parts of the best elite (needs build123d)

  nice -n 19 python tools/make_readme_figures.py
  nice -n 19 python tools/make_readme_figures.py --only gait,parts
  python tools/make_readme_figures.py --archive checkpoints/archive.pkl --out docs/img

Rendering is offscreen (mujoco.Renderer) and small. The output depends only on
the archive; --seed just breaks fitness ties when choosing elites.
"""

import argparse
import os
import struct
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import matplotlib   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import mujoco   # noqa: E402
import numpy as np   # noqa: E402

import plot_archive   # noqa: E402
from replay import open_archive   # noqa: E402

from darwinbot.fitness import UNSTABLE_SCORE   # noqa: E402
from darwinbot import morphology as morpho   # noqa: E402
from darwinbot.evolve import cell_name   # noqa: E402
from darwinbot.sim import Episode   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402

FIGURES = ("robots", "gait", "map", "progress", "parts")
SNIPPETS = {
    "robots": "![Evolved body plans](docs/img/robots.png)",
    "gait": "![Best wheel-less gait](docs/img/gait.gif)",
    "map": "![MAP-Elites archive](docs/img/archive_map.png)",
    "progress": "![Training progress](docs/img/progress.png)",
    "parts": "![Printable parts of the best robot](docs/img/parts.png)",
}
FILES = {"robots": "robots.png", "gait": "gait.gif", "map": "archive_map.png",
         "progress": "progress.png", "parts": "parts.png"}


# ---------------------------------------------------------------- selection

def ranked(archive, seed):
    """(cell, elite) pairs, best first; ties broken by a seeded shuffle.

    Elites at UNSTABLE_SCORE are left out: they are bodies that cannot be
    built (overlapping parts) or whose physics blew up, not robots to show.
    """
    items = [kv for kv in archive.cells.items() if kv[1].fitness > UNSTABLE_SCORE]
    if not items:
        raise SystemExit("no buildable robot in the archive yet")
    order = np.random.default_rng(seed).permutation(len(items))
    items = [items[i] for i in order]
    return sorted(items, key=lambda kv: -kv[1].fitness)


def pick_distinct(items, n=6):
    """
    Best elite of each (leg count, has wheels) group. The first pass takes the
    best group per leg count so every leg count shows up; the rest fill up by
    fitness.
    """
    groups = {}
    for cell, elite in items:                  # items are sorted: first = best
        groups.setdefault((cell[0], cell[1] > 0), (cell, elite))
    firsts, seen = [], set()
    for key, val in groups.items():
        if key[0] not in seen:
            seen.add(key[0])
            firsts.append(val)
    rest = [v for v in groups.values() if v not in firsts]
    chosen = (firsts + rest)[:n]
    return sorted(chosen, key=lambda kv: -kv[1].fitness)


def best_wheelless(items):
    for cell, elite in items:
        if cell[1] == 0:
            return cell, elite
    return items[0]


# ---------------------------------------------------------------- rendering

def follow_camera(episode, azimuth, elevation, distance_scale=3.0):
    """Free camera aimed at the torso; call set_target(cam) before each frame."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = azimuth, elevation
    m = episode.morph
    reach = max(p.reach for p in m.pairs) + m.body_length / 2
    cam.distance = max(0.55, distance_scale * reach)
    return cam


def aim(cam, episode):
    cam.lookat[:] = episode.data.xpos[episode.torso] * np.array([1, 1, 0]) + [0, 0, 0.06]


def settle(episode, ctrl, seconds):
    cpg = episode.reset(ctrl)
    while episode.data.time < seconds:
        episode.step(cpg)
    return cpg


def fmt_size(path):
    return f"{os.path.getsize(path) / 1024:.0f} KB"


def make_robots(items, out, seed):
    chosen = pick_distinct(items)
    flat = get_terrains("flat")[0]
    w, h = 400, 300
    tiles = []
    for cell, elite in chosen:
        ep = Episode(morpho.decode(elite.morph), flat)
        settle(ep, elite.ctrl, 1.5)
        cam = follow_camera(ep, azimuth=135, elevation=-22)
        aim(cam, ep)
        r = mujoco.Renderer(ep.model, height=h, width=w)
        r.update_scene(ep.data, camera=cam)
        tiles.append((r.render().copy(), f"{cell_name(cell)}\nfitness {elite.fitness:.2f}"))
        r.close()

    cols = 3
    rows = -(-len(tiles) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3.45), dpi=100,
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, caption) in zip(axes.flat, tiles):
        ax.imshow(img)
        ax.set_title(caption, fontsize=10)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.005,
                        wspace=0.01, hspace=0.18)
    path = os.path.join(out, FILES["robots"])
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]


def make_gait(items, out, seed, seconds=4.0, fps=15, size=(480, 320)):
    from PIL import Image

    cell, elite = best_wheelless(items)
    ep = Episode(morpho.decode(elite.morph), get_terrains("flat")[0])
    cam = follow_camera(ep, azimuth=90, elevation=-25, distance_scale=2.6)
    r = mujoco.Renderer(ep.model, height=size[1], width=size[0])
    cpg = ep.reset(elite.ctrl)
    frames = []
    for k in range(int(seconds * fps)):
        while ep.data.time < k / fps:
            ep.step(cpg)
        aim(cam, ep)
        r.update_scene(ep.data, camera=cam)
        frames.append(Image.fromarray(r.render().copy()))
    r.close()

    # One shared palette (built from a mosaic of frames) avoids flicker.
    sample = frames[:: max(1, len(frames) // 6)][:6]
    mosaic = Image.new("RGB", (size[0] * len(sample), size[1]))
    for i, f in enumerate(sample):
        mosaic.paste(f, (i * size[0], 0))
    palette = mosaic.quantize(colors=64, method=Image.Quantize.MEDIANCUT,
                              dither=Image.Dither.NONE)
    paletted = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path = os.path.join(out, FILES["gait"])
    paletted[0].save(path, save_all=True, append_images=paletted[1:], loop=0,
                     duration=round(1000 / fps), optimize=True, disposal=1)
    print(f"gait: {cell_name(cell)}, fitness {elite.fitness:.2f}")
    return [path]


def make_plots(archive, out, which):
    paths = []
    for name, fn in (("map", plot_archive.heatmap), ("progress", plot_archive.history)):
        if name not in which:
            continue
        fig = fn(archive)
        if fig is None:
            continue
        path = os.path.join(out, FILES[name])
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


# -------------------------------------------------------------------- parts

def stl_bounds(path):
    """Min and max corner of a binary STL (numpy only)."""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.dtype(
            [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
    pts = data["v"].reshape(-1, 3)
    return pts.min(axis=0), pts.max(axis=0)


def project(cam, point, fovy, w, h):
    """World point -> pixel for a free mujoco camera (azimuth/elevation/distance)."""
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    right = np.cross(fwd, up)
    d = np.asarray(point, dtype=float) - (cam.lookat - cam.distance * fwd)
    f = (h / 2) / np.tan(np.radians(fovy) / 2)
    depth = d @ fwd
    return w / 2 + f * (d @ right) / depth, h / 2 - f * (d @ up) / depth


def make_parts(items, out, seed):
    try:
        from darwinbot import export as dexport
        import build123d   # noqa: F401
    except ImportError:
        print("parts: build123d is not installed (pip install 'darwinbot[cad]'), skipped")
        return []

    cell, elite = items[0]
    morph = morpho.decode(elite.morph)
    with tempfile.TemporaryDirectory() as tmp:
        dexport._write_stls(morph, tmp)
        folder = os.path.join(tmp, "parts")
        names = sorted(os.listdir(folder))
        bounds = {n: stl_bounds(os.path.join(folder, n)) for n in names}

        # Pack parts left to right in rows no wider than max_row mm.
        gap, max_row = 25.0, 520.0
        placed, x, y, row_h = [], 0.0, 0.0, 0.0
        for n in sorted(names, key=lambda n: -(bounds[n][1] - bounds[n][0])[0]):
            lo, hi = bounds[n]
            size = hi - lo
            if x > 0 and x + size[0] > max_row:
                x, y, row_h = 0.0, y + row_h + gap + 18, 0.0
            placed.append((n, x + size[0] / 2, y + size[1] / 2, lo, hi))
            x += size[0] + gap
            row_h = max(row_h, size[1])
        total_w = max(p[1] + (p[4][0] - p[3][0]) / 2 for p in placed)
        total_h = max(p[2] + (p[4][1] - p[3][1]) / 2 for p in placed)
        cx, cy = total_w / 2, total_h / 2

        assets = "".join(
            f'<mesh name="{n[:-4]}" file="{os.path.join(folder, n)}" scale="0.001 0.001 0.001"/>'
            for n, *_ in placed)
        geoms = ""
        for n, px, py, lo, hi in placed:
            c = (lo + hi) / 2                       # bbox centre in mesh coordinates
            pos = np.array([px - cx, cy - py, 0.0]) * 0.001 - c * 0.001 + [0, 0, (hi - lo)[2] * 0.0005]
            tint = "0.95 0.55 0.20 1" if n.startswith(("foot", "wheel")) else "0.35 0.55 0.85 1"
            geoms += (f'<geom name="{n[:-4]}" type="mesh" mesh="{n[:-4]}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
                      f'rgba="{tint}"/>')
        xml = f"""<mujoco><visual><global offwidth="1280" offheight="720" fovy="20"/>
          <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/></visual>
          <asset>{assets}</asset>
          <worldbody><light pos="0 -0.3 1" dir="0 0.3 -1" diffuse="0.5 0.5 0.5"/>
          <geom type="plane" size="2 2 0.1" rgba="0.93 0.93 0.93 1"/>{geoms}</worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    w, h = 1200, 700
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = 90, -62
    cam.lookat[:] = [0, 0, 0]
    tan_v = np.tan(np.radians(model.vis.global_.fovy) / 2)
    tan_h = tan_v * w / h
    # the view is tilted, so the depth of the layout is foreshortened
    cam.distance = 1.12 * max(total_w * 0.0005 / tan_h, total_h * 0.0005 * 0.9 / tan_v) + 0.03
    r = mujoco.Renderer(model, height=h, width=w)
    r.update_scene(data, camera=cam)
    img = r.render().copy()
    r.close()

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    for n, px, py, lo, hi in placed:
        size = hi - lo
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n[:-4])
        # world centre of the mesh box (MuJoCo recentres meshes on compile)
        centre = data.geom_xpos[i] + data.geom_xmat[i].reshape(3, 3) @ model.geom_aabb[i, :3]
        top = centre + np.array([0.0, size[1] / 2 * 0.001 + 0.014, -size[2] / 2 * 0.001])
        u, v = project(cam, top, model.vis.global_.fovy, w, h)
        ax.text(u, v, f"{n[:-4]}\n{size[0]:.0f} x {size[1]:.0f} mm", ha="center",
                va="bottom", fontsize=8, color="#222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.text(12, 22, f"{cell_name(cell)}: printable parts (one of each; blue PLA, orange TPU or wheel)",
            fontsize=10, color="#222", bbox=dict(fc="white", ec="none", alpha=0.75))
    path = os.path.join(out, FILES["parts"])
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="darwinbot: regenerate the README figures")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--out", default="docs/img")
    p.add_argument("--only", default=None,
                   help=f"comma list of figures to make: {', '.join(FIGURES)}")
    p.add_argument("--seed", type=int, default=0, help="tie-break among equal fitness")
    a = p.parse_args()

    which = list(FIGURES) if a.only is None else [s.strip() for s in a.only.split(",") if s.strip()]
    unknown = [s for s in which if s not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")

    archive = open_archive(a.archive)
    if not archive.cells:
        raise SystemExit(f"{a.archive} has no elites")
    os.makedirs(a.out, exist_ok=True)
    items = ranked(archive, a.seed)

    written, t0 = [], time.time()
    steps = {"robots": lambda: make_robots(items, a.out, a.seed),
             "gait": lambda: make_gait(items, a.out, a.seed),
             "parts": lambda: make_parts(items, a.out, a.seed)}
    for name in which:
        if name in steps:
            written += steps[name]()
    written += make_plots(archive, a.out, which)

    for path in written:
        print(f"wrote {path} ({fmt_size(path)})")
    print(f"done in {time.time() - t0:.0f} s")
    print("\nREADME snippets:")
    for name in which:
        if os.path.join(a.out, FILES[name]) in written:
            print(SNIPPETS[name])


if __name__ == "__main__":
    main()

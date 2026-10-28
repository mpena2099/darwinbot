"""
Render every STL of an exported robot on one contact sheet, to check the
printed parts at a glance before slicing them.

    python tools/parts_sheet.py exports/<type>              # writes exports/<type>/parts.png
    python tools/parts_sheet.py exports/<type> --out sheet.png

Each tile shows one part from the same angle, scaled to fit, with its name and
bounding box in mm. Needs only matplotlib; reads ASCII and binary STL.
"""

import argparse
import glob
import os
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_stl(path: str) -> np.ndarray:
    """Triangles of an STL file, shape (n, 3, 3)."""
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [list(map(float, line.split()[1:4])) for line in data.decode().splitlines()
             if line.strip().startswith("vertex")]
        return np.array(v).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data[84:84 + n * 50], dtype=rec)["v"]


def sheet(robot_dir: str, out: str, cols: int = 4) -> int:
    stls = sorted(glob.glob(os.path.join(robot_dir, "**", "*.stl"), recursive=True))
    if not stls:
        raise SystemExit(f"no STL files under {robot_dir}")
    rows = (len(stls) + cols - 1) // cols
    fig = plt.figure(figsize=(3.2 * cols, 3.0 * rows))
    for i, path in enumerate(stls):
        tri = read_stl(path)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolor="#7a9cc6",
                                             edgecolor="#33333320", linewidth=0.1))
        lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        size = " x ".join(f"{s:.0f}" for s in hi - lo)
        ax.set_title(f"{os.path.basename(path)[:-4]}\n{size} mm", fontsize=8)
        ax.view_init(35, -60)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return len(stls)


def main():
    p = argparse.ArgumentParser(description="darwinbot: one image with every printed part")
    p.add_argument("robot_dir", help="an export folder, e.g. exports/8servos-low-roll")
    p.add_argument("--out", default=None, help="image path (default: <robot_dir>/parts.png)")
    a = p.parse_args()
    out = a.out or os.path.join(a.robot_dir, "parts.png")
    n = sheet(a.robot_dir, out)
    print(f"wrote {out} ({n} parts)")


if __name__ == "__main__":
    main()

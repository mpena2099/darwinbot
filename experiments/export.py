"""
experiments/export.py
---------------------
Write the build folder (control file, STL parts, BOM, assembly notes) for an
evolved robot from an archive checkpoint.

  python experiments/export.py                              # best elite
  python experiments/export.py --type 6legs-2wheels-14servos --out my_robot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from replay import load  # noqa: E402

from darwinbot.evolve import cell_of  # noqa: E402
from darwinbot.export import export_robot  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="darwinbot: export an evolved robot")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--type", dest="cell", default=None,
                   help="robot type from replay.py --list (default: best robot overall)")
    p.add_argument("--out", default=None, help="output folder (default: exports/<type>/)")
    p.add_argument("--organic", action="store_true",
                   help="also write hollowed links and elbow plates (topology "
                        "optimisation) to parts_organic/; slower, unvalidated")
    a = p.parse_args()

    archive, elite = load(a.archive, a.cell)
    name = archive.names()[cell_of(elite.behaviour)]
    out = a.out or os.path.join("exports", name)
    info = export_robot(elite, out, name, payload=archive.payload,
                        organic=a.organic, seconds=archive.seconds)
    print(f"wrote {out}")
    print(f"{name}: {info['servos']} servos, {info['printed_parts']} printed parts")
    for part, ratio in info.get("organic", {}).items():
        print(f"  parts_organic/{part}: {ratio:.0%} of the material")


if __name__ == "__main__":
    main()

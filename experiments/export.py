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

from darwinbot import morphology as morpho  # noqa: E402
from darwinbot.evolve import cell_of  # noqa: E402
from darwinbot.export import export_robot, organic_check  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="darwinbot: export an evolved robot")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains the archive was trained with")
    p.add_argument("--type", dest="cell", default=None,
                   help="robot type from replay.py --list (default: best robot overall)")
    p.add_argument("--out", default=None, help="output folder (default: exports/<type>/)")
    p.add_argument("--organic", action="store_true",
                   help="also write hollowed links (topology "
                        "optimisation) to parts_organic/; slower, unvalidated")
    a = p.parse_args()

    archive, elite = load(a.archive, a.cell, a.plugin)
    name = archive.names()[cell_of(elite.behaviour)]
    out = a.out or os.path.join("exports", name)
    info = export_robot(elite, out, name, payload=archive.payload,
                        organic=a.organic, seconds=archive.seconds)
    print(f"wrote {out}")
    print(f"{name}: {info['servos']} servos, {info['printed_parts']} printed parts")
    for part, ratio in info.get("organic", {}).items():
        print(f"  parts_organic/{part}: {ratio:.0%} of the material")
    if a.organic:
        organic_report(elite, info.get("organic", {}), archive, a.archive, name)


def organic_report(elite, ratios, archive, archive_path, name):
    """Re-simulate the robot with its hollow masses, on the archive's terrains
    and task, and warn when the score drops by more than 10%."""
    solid, hollow = organic_check(morpho.decode(elite.morph), elite.ctrl, ratios,
                                  archive.terrains, archive.seconds, archive.task,
                                  archive.payload, archive.variants)
    change = f" ({(hollow - solid) / solid:+.0%})" if solid > 0 else ""
    print(f"organic check: solid {solid:.3f}, hollow {hollow:.3f}{change}")
    if solid > 0 and hollow < 0.9 * solid:
        print(f"warning: the hollow robot scores more than 10% below the solid one; "
              f"re-tune its controller with:\n  python experiments/polish.py --archive "
              f"{archive_path} --types {name} --organic --out {name}-organic.pkl")


if __name__ == "__main__":
    main()

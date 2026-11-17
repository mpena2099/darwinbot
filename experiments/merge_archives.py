"""
Merge archives of the same setup into one, keeping the best elite of every
cell. Used to join islands: separate runs, for example on several Kaggle
sessions at once, that start from a common archive and are merged every few
batches.

    python experiments/merge_archives.py run1/archive.pkl run2/archive.pkl --out merged.pkl
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darwinbot.evolve import load_archive, merge   # noqa: E402


def main():
    p = argparse.ArgumentParser(description="darwinbot: merge archives (islands)")
    p.add_argument("archives", nargs="+", help="archive.pkl files of the same setup")
    p.add_argument("--out", required=True, help="path of the merged archive")
    a = p.parse_args()
    archives = []
    for path in a.archives:
        archives.append(load_archive(path))
        s = archives[-1].stats()
        print(f"{path}: batch {s['batch']}, cells {s['cells']}, best {s['best']:.2f}, "
              f"QD-score {s['qd_score']:.1f}")
    out = merge(archives)
    with open(a.out + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(a.out + ".tmp", a.out)
    s = out.stats()
    print(f"merged -> {a.out}: batch {s['batch']}, cells {s['cells']}, best {s['best']:.2f}, "
          f"QD-score {s['qd_score']:.1f}")


if __name__ == "__main__":
    main()

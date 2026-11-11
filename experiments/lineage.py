"""
experiments/lineage.py
----------------------
Family tree of the robots in an archive: where each elite came from.

  python experiments/lineage.py --type 12servos-mid-trot   # ancestors of one robot
  python experiments/lineage.py                             # overview plot

For one robot it prints the chain of ancestors (batch, fitness, type) and draws
its fitness along the chain. The overview plots, for every elite, how many
different robot types its ancestors went through against its fitness: Nordmoen
et al. (2021) found that elites with more varied ancestry reach higher fitness,
the "stepping stones" that MAP-Elites keeps and a single-objective search
throws away. Archives saved before lineage was recorded have no family tree.
"""

import argparse
import os

import numpy as np


def chain(archive, elite):
    """Ancestor records from the first ancestor to the elite:
    (batch, fitness, type) tuples."""
    return [(rec[1], rec[2], rec[3]) for _, rec in reversed(archive.ancestors(elite))]


def plot_chain(name, steps):
    import matplotlib.pyplot as plt

    batches = [b for b, _, _ in steps]
    fits = [f for _, f, _ in steps]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(batches, fits, "-o", color="tab:blue")
    previous = None
    for b, f, t in steps:
        if t != previous:              # label only where the type changes
            ax.annotate(t, (b, f), textcoords="offset points", xytext=(4, 6),
                        fontsize=7, rotation=20)
            previous = t
    ax.set_xlabel("batch")
    ax.set_ylabel("fitness")
    ax.set_title(f"ancestors of {name}")
    fig.tight_layout()
    return fig


def overview(archive):
    """(fitness, number of distinct ancestor types, chain length) per elite."""
    rows = []
    for e in archive.cells.values():
        steps = chain(archive, e)
        if steps:
            rows.append((e.fitness, len({t for _, _, t in steps}), len(steps)))
    return np.array(rows)


def plot_overview(rows):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    points = ax.scatter(rows[:, 1], rows[:, 0], c=rows[:, 2], cmap="viridis",
                        edgecolor="k", linewidth=0.4)
    ax.set_xlabel("distinct robot types among ancestors")
    ax.set_ylabel("fitness")
    ax.set_title("stepping stones: varied ancestry against fitness")
    fig.colorbar(points, ax=ax, label="generations")
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description="darwinbot: family tree of the archive")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains the archive was trained with")
    p.add_argument("--type", help="robot type from replay.py --list")
    p.add_argument("--out", default="checkpoints", help="output directory")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from replay import open_archive   # same directory; shared error messages

    archive = open_archive(a.archive, a.plugin)
    if not archive.lineage:
        raise SystemExit("this archive has no lineage records (trained before "
                         "lineage was added); retrain to get a family tree")
    os.makedirs(a.out, exist_ok=True)

    if a.type:
        try:
            cell = archive.find(a.type)
        except ValueError as err:
            raise SystemExit(str(err))
        steps = chain(archive, archive.cells[cell])
        for b, f, t in steps:
            print(f"batch {b:3d}  fitness {f:6.2f}  {t}")
        fig = plot_chain(a.type, steps)
        path = os.path.join(a.out, f"lineage_{a.type}.png")
    else:
        rows = overview(archive)
        if len(rows) == 0:
            print("no elite has a lineage chain yet; train more batches")
            return
        if len(rows) > 2:
            r = np.corrcoef(rows[:, 1], rows[:, 0])[0, 1]
            print(f"{len(rows)} elites; correlation between ancestor variety "
                  f"and fitness: {r:+.2f}")
        fig = plot_overview(rows)
        path = os.path.join(a.out, "lineage_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

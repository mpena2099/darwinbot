"""
experiments/plot_archive.py
---------------------------
Plots from a MAP-Elites archive checkpoint.

  python experiments/plot_archive.py
  python experiments/plot_archive.py --archive checkpoints/archive.pkl --show
"""

import argparse
import os
import pickle

import numpy as np

from darwinbot.fitness import UNSTABLE_SCORE
from darwinbot.sim import HEIGHT_MAX


def load(path):
    from replay import open_archive   # same directory; shared error messages
    return open_archive(path)


def heatmap(archive):
    """Left: elites in (duty, height), coloured by fitness and sized by servo
    count. Right: fitness against cost of transport, rolling robots marked."""
    import matplotlib.pyplot as plt

    if not archive.cells:
        raise SystemExit("archive has no elites to plot")
    # unbuildable bodies (UNSTABLE_SCORE) are not robots and squash the scale
    elites = [e for e in archive.cells.values() if e.fitness > UNSTABLE_SCORE]
    if not elites:
        raise SystemExit("archive has no buildable elites to plot")
    duty, height, servos, wheel, cot = np.array([e.behaviour for e in elites]).T
    fits = np.array([e.fitness for e in elites])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    points = ax.scatter(duty, height, c=fits, s=20 + 14 * (servos - servos.min()),
                        cmap="viridis", edgecolor="k", linewidth=0.4)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, HEIGHT_MAX + 0.05)
    ax.set_xlabel("duty (fraction of time on the ground)")
    ax.set_ylabel("height (torso z / body reach)")
    ax.set_title("archive: one dot per elite, size = servos")
    fig.colorbar(points, ax=ax, label="fitness")

    roll = wheel > 0.5
    ax2.scatter(cot[~roll], fits[~roll], s=24, edgecolor="k", linewidth=0.4, label="legs")
    ax2.scatter(cot[roll], fits[roll], s=24, marker="s", edgecolor="k",
                linewidth=0.4, label="wheels (roll)")
    ax2.set_xscale("log")
    ax2.set_xlabel("cost of transport (lower is more efficient)")
    ax2.set_ylabel("fitness")
    ax2.set_title("speed against efficiency")
    ax2.legend()
    fig.tight_layout()
    return fig


def history(archive):
    import matplotlib.pyplot as plt

    if not archive.history:
        print("archive has no history to plot")
        return None
    evals = [entry["evals"] for entry in archive.history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(evals, [entry["best"] for entry in archive.history], label="best")
    ax.plot(evals, [entry["mean"] for entry in archive.history], label="mean")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("fitness")
    ax2 = ax.twinx()
    ax2.plot(evals, [entry["cells"] for entry in archive.history],
             color="tab:green", linestyle="--", label="cells")
    ax2.set_ylabel("filled cells")
    handles = ax.get_lines() + ax2.get_lines()
    ax.legend(handles, [h.get_label() for h in handles], loc="best")
    ax.set_title("archive progress")
    return fig


def main():
    p = argparse.ArgumentParser(description="darwinbot: plot an archive")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--out", default="checkpoints", help="output directory")
    p.add_argument("--show", action="store_true", help="display instead of Agg")
    a = p.parse_args()

    if not a.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archive = load(a.archive)
    os.makedirs(a.out, exist_ok=True)

    figures = [("archive_heatmap.png", heatmap(archive)),
               ("archive_history.png", history(archive))]
    for name, fig in figures:
        if fig is None:
            continue
        path = os.path.join(a.out, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")

    if a.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

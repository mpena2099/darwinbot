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

from darwinbot import morphology as morpho
from darwinbot.evolve import N_SERVO_BINS, SERVO_MIN, SERVO_STEP


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def heatmap(archive):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if not archive.cells:
        raise SystemExit("archive has no elites to plot")
    n_bins = N_SERVO_BINS
    fits = [elite.fitness for elite in archive.cells.values()]
    norm = Normalize(vmin=min(fits), vmax=max(fits))

    fig, axes = plt.subplots(1, morpho.N_PAIRS, figsize=(4 * morpho.N_PAIRS, 3.4),
                             squeeze=False)
    image = None
    for panel, ax in enumerate(axes[0], start=1):
        grid = np.full((morpho.N_PAIRS + 1, n_bins), np.nan)
        for (pairs, wheeled, leg_bin), elite in archive.cells.items():
            if pairs == panel:
                grid[wheeled, leg_bin] = elite.fitness
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                          norm=norm)
        ax.set_title(f"{panel} leg pair(s)")
        ax.set_xlabel("servos")
        ax.set_ylabel("wheeled pairs")
        ax.set_xticks(range(n_bins), [SERVO_MIN + SERVO_STEP * i for i in range(n_bins)])
        ax.set_yticks(range(morpho.N_PAIRS + 1))
    fig.colorbar(image, ax=axes[0].tolist(), label="fitness")
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

"""
experiments/plot_progress.py
-----------------------------
Real-time plot of evolution progress.

Run this in a SECOND terminal while evolution is running:

    python experiments/plot_progress.py

Or point it at a specific checkpoint directory:

    python experiments/plot_progress.py --checkpoint-dir checkpoints

It polls for new checkpoint files every few seconds and updates the plot.
Close the window to stop.
"""

import os
import glob
import pickle
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from darwinbot.genome import (
    N_LIMBS, GENES_PER_LIMB, G_HAS_WHEEL,
    seg_base, GS_LENGTH, N_SEGMENTS,
)

POLL_INTERVAL = 4.0   # seconds between checkpoint scans


# ── Data loading ──────────────────────────────────────────────────────────────

def load_checkpoints(checkpoint_dir: str) -> list[dict]:
    """
    Load all checkpoint .pkl files in order.
    Returns list of dicts: {gen, max, avg, std, min, n_wheels_avg}
    """
    pattern = os.path.join(checkpoint_dir, "gen_*.pkl")
    files   = sorted(glob.glob(pattern))
    records = []

    for fpath in files:
        try:
            with open(fpath, "rb") as f:
                ckpt = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            continue  # file still being written

        gen  = ckpt["generation"]
        pop  = ckpt["population"]

        fitnesses = [ind.fitness.values[0]
                     for ind in pop if ind.fitness.valid]
        if not fitnesses:
            continue

        # Count average wheels per individual
        wheel_counts = []
        for ind in pop:
            genes  = np.array(ind)
            wheels = sum(
                round(genes[i * GENES_PER_LIMB + G_HAS_WHEEL])
                for i in range(N_LIMBS)
            )
            wheel_counts.append(wheels)

        records.append({
            "gen":           gen,
            "max":           max(fitnesses),
            "avg":           np.mean(fitnesses),
            "std":           np.std(fitnesses),
            "min":           min(fitnesses),
            "n_wheels_avg":  np.mean(wheel_counts),
        })

    return records


def load_hof(checkpoint_dir: str):
    """Return the Hall of Fame from the most recent checkpoint."""
    pattern = os.path.join(checkpoint_dir, "gen_*.pkl")
    files   = sorted(glob.glob(pattern))
    if not files:
        return None
    try:
        with open(files[-1], "rb") as f:
            ckpt = pickle.load(f)
        return ckpt.get("hof")
    except Exception:
        return None


# ── Plot setup ────────────────────────────────────────────────────────────────

def make_figure():
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_fitness  = fig.add_subplot(gs[0, :])   # top row: full width
    ax_wheels   = fig.add_subplot(gs[1, 0])   # bottom left
    ax_hof      = fig.add_subplot(gs[1, 1])   # bottom right

    for ax in (ax_fitness, ax_wheels, ax_hof):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaacc", labelsize=9)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    fig.suptitle("darwinbot — evolution progress",
                 color="#e0e0ff", fontsize=13, fontweight="bold", y=0.98)

    return fig, ax_fitness, ax_wheels, ax_hof


def update_plot(fig, ax_fitness, ax_wheels, ax_hof,
                records: list, hof):

    gens      = [r["gen"]          for r in records]
    maxs      = [r["max"]          for r in records]
    avgs      = [r["avg"]          for r in records]
    stds      = [r["std"]          for r in records]
    mins      = [r["min"]          for r in records]
    n_wheels  = [r["n_wheels_avg"] for r in records]

    avgs_arr = np.array(avgs)
    stds_arr = np.array(stds)

    # ── Fitness curves ────────────────────────────────────────────────────────
    ax_fitness.cla()
    ax_fitness.set_facecolor("#16213e")

    ax_fitness.fill_between(gens,
                            avgs_arr - stds_arr,
                            avgs_arr + stds_arr,
                            alpha=0.18, color="#7b68ee")

    ax_fitness.plot(gens, maxs,  color="#00d4ff", linewidth=2.0,
                    label="max fitness",  zorder=3)
    ax_fitness.plot(gens, avgs,  color="#7b68ee", linewidth=1.5,
                    label="avg fitness",  zorder=2)
    ax_fitness.plot(gens, mins,  color="#ff6b6b", linewidth=1.0,
                    linestyle="--", alpha=0.6, label="min fitness", zorder=1)

    ax_fitness.axhline(0, color="#555577", linewidth=0.8, linestyle=":")

    ax_fitness.set_xlabel("generation")
    ax_fitness.set_ylabel("fitness score")
    ax_fitness.set_title("fitness over generations",
                         color="#c0c0ee", fontsize=10)
    ax_fitness.legend(fontsize=8, facecolor="#1a1a2e",
                      labelcolor="#ddddff", framealpha=0.7)
    ax_fitness.tick_params(colors="#aaaacc")
    ax_fitness.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for spine in ax_fitness.spines.values():
        spine.set_edgecolor("#333355")

    # Annotate best ever
    if maxs:
        best_gen = gens[int(np.argmax(maxs))]
        best_val = max(maxs)
        ax_fitness.annotate(
            f"best: {best_val:.3f}",
            xy=(best_gen, best_val),
            xytext=(best_gen + max(len(gens) * 0.03, 0.5), best_val),
            color="#00d4ff", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#00d4ff", lw=0.8),
        )

    # ── Average wheel count ───────────────────────────────────────────────────
    ax_wheels.cla()
    ax_wheels.set_facecolor("#16213e")
    ax_wheels.plot(gens, n_wheels, color="#ffd700", linewidth=1.8,
                   marker="o", markersize=3)
    ax_wheels.set_ylim(-0.1, N_LIMBS + 0.2)
    ax_wheels.set_yticks(list(range(N_LIMBS + 1)))
    ax_wheels.set_xlabel("generation")
    ax_wheels.set_ylabel("avg wheels / individual")
    ax_wheels.set_title("wheel count evolution",
                        color="#c0c0ee", fontsize=10)
    ax_wheels.tick_params(colors="#aaaacc")
    ax_wheels.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for spine in ax_wheels.spines.values():
        spine.set_edgecolor("#333355")

    # ── Hall of Fame ──────────────────────────────────────────────────────────
    ax_hof.cla()
    ax_hof.set_facecolor("#16213e")
    ax_hof.axis("off")
    ax_hof.set_title("hall of fame (top 5)",
                     color="#c0c0ee", fontsize=10)

    if hof:
        rows = []
        for rank, ind in enumerate(hof):
            genes = np.array(ind)
            fit   = ind.fitness.values[0]
            n_w     = sum(round(genes[i * GENES_PER_LIMB + G_HAS_WHEEL]) for i in range(N_LIMBS))
            avg_len = np.mean([
                genes[i * GENES_PER_LIMB + seg_base(0) + GS_LENGTH]
                for i in range(N_LIMBS)
            ])
            rows.append([f"#{rank+1}", f"{fit:.3f}",
                         str(n_w), f"{avg_len:.2f}m"])

        col_labels = ["rank", "fitness", "wheels", "avg len"]
        table = ax_hof.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor("#1e1e3e" if row % 2 == 0 else "#16213e")
            cell.set_edgecolor("#333355")
            cell.set_text_props(color="#ddddff")
            if row == 0:
                cell.set_facecolor("#2a2a5e")
                cell.set_text_props(color="#ffffff", fontweight="bold")

    fig.canvas.draw()
    fig.canvas.flush_events()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="darwinbot real-time evolution plotter")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory with gen_XXXX.pkl files")
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL,
                        help="Polling interval in seconds (default: 4)")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        print(f"[plot] Waiting for checkpoint directory: {checkpoint_dir}")

    plt.ion()
    fig, ax_fitness, ax_wheels, ax_hof = make_figure()
    plt.show()

    last_n = 0

    print(f"[plot] Watching: {os.path.abspath(checkpoint_dir)}")
    print(f"[plot] Polling every {args.interval}s — close window to stop")

    while plt.fignum_exists(fig.number):
        if os.path.isdir(checkpoint_dir):
            records = load_checkpoints(checkpoint_dir)
            hof     = load_hof(checkpoint_dir)

            if len(records) != last_n:
                update_plot(fig, ax_fitness, ax_wheels, ax_hof, records, hof)
                last_n = len(records)
                gen = records[-1]["gen"] if records else 0
                best = max(r["max"] for r in records) if records else 0
                print(f"[plot] gen={gen:4d}  best_fitness={best:.4f}")
            else:
                # No new data — just keep the window alive
                fig.canvas.flush_events()

        time.sleep(args.interval)

    print("[plot] Window closed.")


if __name__ == "__main__":
    main()

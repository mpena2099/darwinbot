"""
tools/make_guide_figures.py
---------------------------
Draw the illustrations used by the evolution guide, in Portuguese and English.

  docs/guia-evolucao.md      cycle_pt, genome_pt, loops_pt, cmaes_pt, map_pt, randomize_pt
  docs/evolution-guide.md    the same six figures with the _en suffix

  python tools/make_guide_figures.py
  python tools/make_guide_figures.py --out /tmp/guide

Pure matplotlib and numpy (Agg backend): no MuJoCo, no archive, a few seconds.
Every figure is 900x500 px at dpi 100, white background, flat colours.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch   # noqa: E402

INK = "#1f2933"
MUTED = "#6b7280"
PAPER = "#ffffff"
GRID = "#d7dce3"
BLUE = "#2f6fed"
AMBER = "#c9861f"
GREEN = "#2fa36b"
VIOLET = "#7c5cd6"
RED = "#d94a4a"
TINT = {"blue": "#eaf1fe", "amber": "#fdf3e2", "green": "#e8f6ef", "violet": "#f0ebfd"}
BLOCK_COLOURS = [AMBER, BLUE, VIOLET, GREEN]

COPY = {
    "pt": {
        "cycle": {"steps": [("população", "robôs guardados"), ("variação", "mutação"),
                            ("avaliação", "simulação"), ("seleção", "arquivo")],
                  "heredity": "herança", "center": "uma geração"},
        "genome": {"head": "genoma: números de 0 a 1", "body": "corpo decodificado",
                   "blocks": ["núcleo", "junta 1", "junta 2", "roda"],
                   "genes": {"active": "ativo", "parent": "pai", "slot": "encaixe",
                             "type": "tipo", "axis": "eixo", "length": "comprimento"},
                   "torso": "tronco", "joint": "junta", "foot": "pé", "wheel": "roda"},
        "loops": {"outer": "MAP-Elites nos corpos", "inner": "CMA-ES no controle",
                  "pick": "escolher\npai", "mutate": "mutar\ncorpo", "insert": "inserir\nno mapa",
                  "sample": "sortear\ncontroles", "simulate": "simular", "update": "atualizar",
                  "evaluate": "avaliar cada corpo", "repeat": "próxima geração"},
        "cmaes": {"title": "CMA-ES: a nuvem se move e encolhe",
                  "x": "parâmetro da junta 1", "y": "parâmetro da junta 2",
                  "gen": "geração", "optimum": "ótimo"},
        "map": {"left": "Um vencedor só", "right": "MAP-Elites",
                "note_left": "todos convergem", "note_right": "uma elite por célula",
                "x": "tempo no chão (duty)", "y": "altura do tronco", "fitness": "nota"},
        "randomize": {"labels": ["atrito -30%", "massa +10%", "torque -15%", "latência 2 ticks"],
                      "head": "mesmo robô, físicas diferentes",
                      "caption": "treinado em todas elas"},
    },
    "en": {
        "cycle": {"steps": [("population", "saved robots"), ("variation", "mutation"),
                            ("evaluation", "simulation"), ("selection", "archive")],
                  "heredity": "heredity", "center": "one generation"},
        "genome": {"head": "genome: numbers from 0 to 1", "body": "decoded body",
                   "blocks": ["core", "joint 1", "joint 2", "wheel"],
                   "genes": {"active": "active", "parent": "parent", "slot": "slot",
                             "type": "type", "axis": "axis", "length": "length"},
                   "torso": "torso", "joint": "joint", "foot": "foot", "wheel": "wheel"},
        "loops": {"outer": "MAP-Elites over bodies", "inner": "CMA-ES over the controller",
                  "pick": "pick\nparent", "mutate": "mutate\nbody", "insert": "insert\ninto map",
                  "sample": "sample\ncontrollers", "simulate": "simulate", "update": "update",
                  "evaluate": "evaluate each body", "repeat": "next generation"},
        "cmaes": {"title": "CMA-ES: the cloud moves and shrinks",
                  "x": "joint 1 parameter", "y": "joint 2 parameter",
                  "gen": "generation", "optimum": "optimum"},
        "map": {"left": "Single winner", "right": "MAP-Elites",
                "note_left": "everyone converges", "note_right": "one elite per cell",
                "x": "time on the ground (duty)", "y": "torso height", "fitness": "fitness"},
        "randomize": {"labels": ["friction -30%", "mass +10%", "torque -15%",
                                 "latency 2 ticks"],
                      "head": "same robot, different floors",
                      "caption": "trained on all of them"},
    },
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "legend.frameon": False,
})

GENES = [
    (0, "active", 0.85), (0, "parent", 0.30), (0, "slot", 0.55), (0, "type", 0.75),
    (1, "active", 0.65), (1, "parent", 0.40), (1, "slot", 0.60), (1, "type", 0.80),
    (2, "active", 0.55), (2, "axis", 0.70), (2, "length", 0.45),
    (3, "type", 0.90), (3, "length", 0.35),
]


# ------------------------------------------------------------------ helpers

def canvas():
    fig = plt.figure(figsize=(9, 5), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def card(ax, cx, cy, w, h, title, sub=None, fc="white", ec=GRID, lw=1.8, fs=11):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.015",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        ax.text(cx, cy + 0.016, title, ha="center", va="center", fontsize=fs,
                weight="bold", zorder=4)
        ax.text(cx, cy - 0.024, sub, ha="center", va="center", fontsize=8.5,
                color=MUTED, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fs, zorder=4,
                linespacing=1.3)


def arrow(ax, p, q, color=INK, rad=0.0, lw=2.0, ms=13):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def edge(c, other, hw, hh):
    d = np.subtract(other, c)
    t = min(hw / abs(d[0]) if d[0] else np.inf,
            hh / abs(d[1]) if d[1] else np.inf)
    return tuple(np.asarray(c) + d * t)


def wheel(ax, cx, cy, r, fc, ec, lw=1.8):
    # the canvas is 9x5, so the width needs the 5/9 factor to look circular
    ax.add_patch(Ellipse((cx, cy), width=2 * r * 5 / 9, height=2 * r,
                         fc=fc, ec=ec, lw=lw, zorder=3))


# ------------------------------------------------------------------- cycle

def draw_cycle(fig, ax, L):
    spots = [(0.50, 0.86), (0.845, 0.50), (0.50, 0.14), (0.155, 0.50)]
    tints = ["blue", "amber", "green", "violet"]
    edges = [BLUE, AMBER, GREEN, VIOLET]
    for (cx, cy), tint, ec, (title, sub) in zip(spots, tints, edges, L["steps"]):
        card(ax, cx, cy, 0.24, 0.13, title, sub, fc=TINT[tint], ec=ec)
    hw, hh = 0.12, 0.065
    for i in range(4):
        p, q = spots[i], spots[(i + 1) % 4]
        arrow(ax, edge(p, q, hw, hh), edge(q, p, hw, hh), color=MUTED, lw=2.2)
    ax.text(0.284, 0.722, L["heredity"], rotation=46, fontsize=10, style="italic",
            ha="center", va="center")
    ax.text(0.5, 0.5, L["center"], ha="center", va="center", fontsize=10,
            color=MUTED, style="italic")


# ------------------------------------------------------------------ genome

def draw_genome(fig, ax, L):
    ax.text(0.02, 0.94, L["head"], fontsize=11, weight="bold", va="center")
    ax.text(0.98, 0.94, L["body"], fontsize=11, weight="bold", va="center", ha="right")
    top, h, step, x, w = 0.865, 0.052, 0.058, 0.10, 0.24
    for i, (block, gene, value) in enumerate(GENES):
        y, col = top - i * step - h, BLOCK_COLOURS[block]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                    fc=col, ec=col, lw=1, alpha=0.92, zorder=3))
        ax.text(x + 0.012, y + h / 2, L["genes"][gene], fontsize=7.2, color="white",
                va="center", zorder=4)
        ax.text(x + w - 0.008, y + h / 2, f"{value:.2f}", fontsize=6.6, color="white",
                va="center", ha="right", zorder=4)
    for (first, last), name, col in zip([(0, 3), (4, 7), (8, 10), (11, 12)],
                                        L["blocks"], BLOCK_COLOURS):
        yc = top - (first + last) / 2 * step - h / 2
        ax.text(0.073, yc, name, rotation=90, ha="center", va="center", fontsize=8.4,
                color=col, weight="bold")
        ax.plot([0.087, 0.087], [top - first * step, top - (last + 1) * step + 0.006],
                color=col, lw=2.4, alpha=0.5)

    torso, left, right = (0.72, 0.70), (0.63, 0.48), (0.81, 0.48)
    foot, hub = (0.63, 0.30), (0.81, 0.30)
    ax.plot([0.665, 0.665, left[0]], [0.65, 0.56, 0.52], color=INK, lw=1.8, zorder=1)
    ax.plot([0.775, 0.775, right[0]], [0.65, 0.56, 0.52], color=INK, lw=1.8, zorder=1)
    ax.plot([0.63, 0.63], [0.44, 0.33], color=INK, lw=1.8, zorder=1)
    ax.plot([0.81, 0.81], [0.44, 0.345], color=INK, lw=1.8, zorder=1)
    card(ax, *torso, 0.18, 0.10, L["torso"], fc=TINT["amber"], ec=AMBER, fs=9.5)
    card(ax, *left, 0.10, 0.08, L["joint"], fc=TINT["blue"], ec=BLUE, fs=8.5)
    card(ax, *right, 0.10, 0.08, L["joint"], fc=TINT["blue"], ec=BLUE, fs=8.5)
    card(ax, *foot, 0.09, 0.06, L["foot"], fc=TINT["green"], ec=GREEN, fs=8.5)
    wheel(ax, *hub, 0.045, TINT["green"], GREEN)
    ax.text(*hub, L["wheel"], fontsize=7.5, ha="center", va="center", zorder=4)

    starts = [0.752, 0.520, 0.317, 0.172]
    ends = [(0.628, 0.715), (left[0] - 0.05, 0.49), (right[0] - 0.05, 0.49),
            (hub[0] - 0.025, 0.315)]
    for y0, end, col in zip(starts, ends, BLOCK_COLOURS):
        arrow(ax, (0.35, y0), end, color=col, lw=1.8, rad=-0.08, ms=11)


# ------------------------------------------------------------------- loops

def draw_loops(fig, ax, L):
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc=TINT["blue"], ec=BLUE, lw=1.8, zorder=0))
    ax.text(0.045, 0.90, L["outer"], fontsize=11, weight="bold", color=BLUE, va="center")
    top = 0.68
    for cx, key in ((0.14, "pick"), (0.42, "mutate"), (0.86, "insert")):
        card(ax, cx, top + 0.06, 0.20, 0.12, L[key])
    arrow(ax, (0.24, top + 0.06), (0.32, top + 0.06), color=MUTED)
    arrow(ax, (0.86, top + 0.12), (0.14, top + 0.12), color=BLUE, rad=0.18, lw=1.8, ms=11)

    ax.add_patch(FancyBboxPatch((0.07, 0.13), 0.62, 0.44,
                                boxstyle="round,pad=0,rounding_size=0.018",
                                fc=TINT["amber"], ec=AMBER, lw=1.8, zorder=0))
    ax.text(0.09, 0.525, L["inner"], fontsize=10.5, weight="bold", color=AMBER, va="center")
    row = 0.315
    for cx, key in ((0.18, "sample"), (0.38, "simulate"), (0.58, "update")):
        card(ax, cx, row + 0.05, 0.18, 0.10, L[key], fs=9.5)
    arrow(ax, (0.27, row + 0.05), (0.29, row + 0.05), color=AMBER)
    arrow(ax, (0.47, row + 0.05), (0.49, row + 0.05), color=AMBER)
    arrow(ax, (0.58, row), (0.18, row), color=AMBER, rad=-0.45, lw=1.8, ms=11)
    ax.text(0.38, 0.195, L["repeat"], fontsize=8.5, color=MUTED, style="italic",
            ha="center", va="center")
    arrow(ax, (0.42, top), (0.42, 0.57), color=AMBER, lw=2.2)
    ax.text(0.44, 0.625, L["evaluate"], fontsize=8.5, color=MUTED, va="center")
    arrow(ax, (0.69, 0.45), (0.80, top), color=AMBER, rad=-0.25, lw=1.8, ms=11)


# ------------------------------------------------------------------- cmaes

ANG = np.deg2rad(35)


def valley(p):
    x, y = p[..., 0], p[..., 1]
    u = x * np.cos(ANG) + y * np.sin(ANG)
    v = -x * np.sin(ANG) + y * np.cos(ANG)
    return (u / 2.6) ** 2 + (v / 0.55) ** 2


def cma_snapshots(seed=0, gens=7, shots=(0, 2, 4, 6), lam=40, mu=20):
    """Simplified covariance adaptation, in the spirit of CMA-ES."""
    rng = np.random.default_rng(seed)
    m, C = np.array([2.7, 1.9]), np.eye(2) * 0.8 ** 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    out = []
    for g in range(gens):
        x = m + rng.standard_normal((lam, 2)) @ np.linalg.cholesky(C).T
        if g in shots:
            out.append((x, C.copy()))
        elite = x[np.argsort(valley(x))[:mu]]
        m = w @ elite
        d = elite - m
        C = 1.8 * (d.T * w) @ d + 1e-3 * np.eye(2)  # inflate so it does not collapse
    return out


def draw_cmaes(fig, ax, L):
    ax.remove()
    ax = fig.add_axes([0.13, 0.16, 0.84, 0.74])
    g = np.linspace(-1.2, 4.2, 220)
    X, Y = np.meshgrid(g, g)
    F = valley(np.dstack([X, Y]))
    levels = [0.1, 0.3, 0.8, 1.8, 4, 9, 20, 45]
    ax.contourf(X, Y, F, levels=levels, cmap="Greys", alpha=0.15, zorder=0)
    ax.contour(X, Y, F, levels=levels[1:-1], colors=GRID, linewidths=0.9, zorder=1)
    ax.plot(0, 0, marker="*", ms=17, color=AMBER, mec=INK, mew=0.8, zorder=6,
            label=L["optimum"])
    for i, (x, C) in enumerate(cma_snapshots()):
        vals, vecs = np.linalg.eigh(C)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ax.scatter(x[:, 0], x[:, 1], s=18, color=["#9dbbf2", "#5b8def", VIOLET, RED][i],
                   alpha=0.9, linewidths=0, zorder=3, label=f'{L["gen"]} {i + 1}')
        ax.add_patch(Ellipse(x.mean(axis=0), 4 * np.sqrt(vals[0]), 4 * np.sqrt(vals[1]),
                             angle=ang, fill=False, lw=2, zorder=4,
                             ec=["#9dbbf2", "#5b8def", VIOLET, RED][i]))
    ax.set(xlim=(-1.2, 4.2), ylim=(-1.2, 4.2), xlabel=L["x"], ylabel=L["y"])
    ax.set_title(L["title"], fontsize=12, color=INK, pad=10)
    ax.legend(loc="upper left", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------- map

def draw_map(fig, ax, L):
    ax.remove()
    left = fig.add_axes([0.07, 0.16, 0.35, 0.70])
    right = fig.add_axes([0.56, 0.16, 0.35, 0.70])
    cax = fig.add_axes([0.94, 0.16, 0.015, 0.70])
    rng = np.random.default_rng(4)
    pts = rng.normal([0.42, 0.55], [0.055, 0.05], size=(70, 2))
    left.scatter(pts[:, 0], pts[:, 1], s=22, color="#6f9ae8", alpha=0.75, linewidths=0)
    left.plot(0.42, 0.55, marker="*", ms=16, color=AMBER, mec=INK, mew=0.8)
    left.set_title(L["left"], fontsize=12, color=INK, pad=10)

    fit = rng.random((5, 6))
    fit[rng.random((5, 6)) < 0.22] = np.nan
    mesh = right.pcolormesh(np.linspace(0, 1, 7), np.linspace(0, 1, 6),
                            np.ma.masked_invalid(fit), cmap="viridis", vmin=0, vmax=1,
                            edgecolors="white", linewidth=1.4)
    right.set_facecolor("#eef0f3")
    right.set_title(L["right"], fontsize=12, color=INK, pad=10)

    for a, note in ((left, L["note_left"]), (right, L["note_right"])):
        a.set(xlim=(0, 1), ylim=(0, 1), xlabel=L["x"], ylabel=L["y"])
        a.set_xticks([])
        a.set_yticks([])
        a.text(0.03, 0.97, note, transform=a.transAxes, va="top", fontsize=8.5,
               style="italic", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label(L["fitness"], fontsize=9.5)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(labelsize=8)


# --------------------------------------------------------------- randomize

def sketch(ax, cx, gy):
    ax.add_patch(FancyBboxPatch((cx - 0.055, gy + 0.085), 0.11, 0.06,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                fc=TINT["amber"], ec=AMBER, lw=1.6, zorder=3))
    for dx in (-0.028, 0.028):
        ax.plot([cx + dx, cx + dx * 1.25, cx + dx * 1.15],
                [gy + 0.085, gy + 0.045, gy + 0.003], color=BLUE, lw=2.6,
                solid_capstyle="round", zorder=2)
    ax.plot([cx + 0.05, cx + 0.068], [gy + 0.10, gy + 0.065], color=INK, lw=1.6, zorder=2)
    wheel(ax, cx + 0.068, gy + 0.033, 0.033, TINT["green"], GREEN)


def draw_randomize(fig, ax, L):
    ax.text(0.5, 0.92, L["head"], ha="center", va="center", fontsize=10.5,
            color=MUTED, style="italic")
    shades = ["#8a94a3", "#9b8f7a", "#7f9488", "#8390a8"]
    for cx, gy, label, shade in zip([0.145, 0.385, 0.625, 0.865],
                                    [0.28, 0.26, 0.30, 0.27], L["labels"], shades):
        sketch(ax, cx, gy)
        ax.plot([cx - 0.10, cx + 0.10], [gy, gy], color=shade, lw=2.2,
                solid_capstyle="round")
        for k in range(5):
            x0 = cx - 0.095 + k * 0.05
            ax.plot([x0, x0 + 0.016], [gy, gy - 0.016], color=shade, lw=1, alpha=0.55)
        ax.text(cx, 0.58, label, ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(0.5, 0.10, L["caption"], ha="center", va="center", fontsize=12.5,
            style="italic", color=INK)


# --------------------------------------------------------------------- main

FIGURES = {"cycle": draw_cycle, "genome": draw_genome, "loops": draw_loops,
           "cmaes": draw_cmaes, "map": draw_map, "randomize": draw_randomize}


def main():
    p = argparse.ArgumentParser(description="darwinbot: draw the evolution guide figures")
    p.add_argument("--out", default="docs/img/guide", help="output directory")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for lang, labels in COPY.items():
        for name, draw in FIGURES.items():
            fig, ax = canvas()
            draw(fig, ax, labels[name])
            path = os.path.join(a.out, f"{name}_{lang}.png")
            fig.savefig(path, dpi=100, facecolor=PAPER)
            plt.close(fig)
            print("wrote", path)


if __name__ == "__main__":
    main()

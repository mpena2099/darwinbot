"""
Offline analysis of a DARWINBOT_TRACE directory (numpy only).

    python tools/speed/analyse_trace.py <trace dir> [--out report.md]

Sections: acceptance and compute by origin, L1 (race against the incumbent),
L6 (reduced evaluation designs), L7 (budgets), L2 (potential of a staged search
on the command structure). The layout of the files is in darwinbot/trace.py.

Caveat for L1: the incumbent is the elite of the cell the child finally fell
in, which a live rule only knows after the run; a live rule has to guess the
cell (for instance from the parent).
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from darwinbot import fitness as fit          # noqa: E402
from darwinbot.sim import Metrics             # noqa: E402

CHECKPOINTS = (0.1, 0.2, 0.3, 0.5)
MARGINS = (0.0, 0.02, 0.05, 0.1)
ORIGINS = ("mutate", "graft", "same", "random")


# ---------------------------------------------------------------- loading

class Body:
    """One traced body: the jsonl record and, when present, its npz arrays."""

    def __init__(self, rec, z):
        self.rec = rec
        self.z = z
        self.origin = rec.get("origin") or "random"
        self.before = rec.get("before")
        self.accepted = bool(rec.get("accepted"))
        self.fitness = None if z is None else np.asarray(z["fitness"], float)
        self._objs = None

    @property
    def n(self):
        return 0 if self.fitness is None else len(self.fitness)

    def metrics(self):
        """Metrics objects as [eval][terrain][variant][command]."""
        if self._objs is None:
            arr = self.z["metrics"]
            names = [str(f) for f in self.z["fields"]]
            self._objs = {}
            self._arr, self._names = arr, names
        return self._objs

    def m(self, e, t, v, c):
        key = (e, t, v, c)
        objs = self.metrics()
        if key not in objs:
            row = dict(zip(self._names, (float(x) for x in self._arr[e, t, v, c])))
            row["fell"] = bool(row["fell"])
            row["unstable"] = bool(row["unstable"])
            objs[key] = Metrics(**row)
        return objs[key]


def load(directory):
    bodies = []
    with open(os.path.join(directory, "bodies.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            z = None
            if rec.get("file"):
                p = os.path.join(directory, rec["file"])
                if os.path.exists(p):
                    z = dict(np.load(p))
            bodies.append(Body(rec, z))
    return bodies


# ---------------------------------------------------------------- helpers

def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def num(x, d=3):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def aggregate_cols(s):
    """fitness.aggregate over the last axis of an array."""
    s = np.asarray(s, float)
    return 0.7 * s.mean(axis=-1) + 0.3 * s.min(axis=-1)


def rank(x):
    """Ranks with ties averaged."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        idx = x == v
        if idx.sum() > 1:
            r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- general

def section_general(bodies):
    out = ["## Acceptance and compute by origin", ""]
    rows = []
    tot_evals = sum(b.rec["evals"] for b in bodies) or 1
    tot_sec = sum(b.rec.get("seconds") or 0 for b in bodies) or 1
    for o in ORIGINS + ("all",):
        bs = [b for b in bodies if o == "all" or b.origin == o]
        if not bs:
            continue
        ev = sum(b.rec["evals"] for b in bs)
        sec = sum(b.rec.get("seconds") or 0 for b in bs)
        acc = sum(b.accepted for b in bs)
        rows.append([o, len(bs), acc, pct(acc / len(bs)), ev, pct(ev / tot_evals),
                     f"{sec:.0f}", pct(sec / tot_sec),
                     num(ev / sec if sec else float("nan"), 2),
                     num(sec / acc if acc else float("nan"), 0)])
    out.append(md_table(["origin", "bodies", "accepted", "rate", "evals", "share evals",
                         "wall s", "share wall", "evals/s", "wall s per accepted"], rows))
    nb = sum(1 for b in bodies if b.z is None)
    if nb:
        out += ["", f"{nb} bodies have no npz (unbuildable or failed)."]
    return out


# ---------------------------------------------------------------- L1

def race_rows(bodies):
    """Rows of the race table for a set of children with an incumbent."""
    rows = []
    tot = sum(b.n for b in bodies)
    acc = [b for b in bodies if b.accepted]
    for rule in ("best<inc-margin", "best+slope*rest<inc-margin"):
        for margin in MARGINS if rule.startswith("best<") else (0.0, 0.05):
            for frac in CHECKPOINTS:
                saved, fr = 0, 0
                for b in bodies:
                    n = b.n
                    k = max(int(math.ceil(frac * n)), 2)
                    if k >= n:
                        continue
                    best = np.maximum.accumulate(b.fitness)
                    proj = best[k - 1]
                    if rule.startswith("best+"):
                        w = max(k // 3, 1)
                        slope = (best[k - 1] - best[k - 1 - w]) / w
                        proj = proj + slope * (n - k)
                    if proj + margin < b.before:
                        saved += n - k
                        fr += b.accepted
                rows.append([rule, margin, f"{int(100 * frac)}%", pct(saved / tot if tot else float("nan")),
                             f"{fr}/{len(acc)}", pct(fr / len(acc) if acc else float("nan"))])
    return rows


def section_l1(bodies):
    out = ["## L1: race against the incumbent of the final cell", "",
           "Rule stops a child at a checkpoint (share of its evaluations) when its best so far "
           "(plus the projected gain, for the slope rule) plus the margin is still below the "
           "incumbent. `saved` = share of the evaluations of that group not run; "
           "`false rejects` = accepted children the rule would have stopped.", ""]
    header = ["rule", "margin", "checkpoint", "saved", "false rejects", "share of accepted"]
    occupied = [b for b in bodies if b.z is not None and b.before is not None and b.n > 3]
    empty = [b for b in bodies if b.z is not None and b.before is None and b.n > 3]
    out += ["### Occupied final cell, all origins", "",
            f"{len(occupied)} children, {sum(b.accepted for b in occupied)} accepted.", "",
            md_table(header, race_rows(occupied)) if occupied else "no data", ""]
    for o in ORIGINS:
        sub = [b for b in occupied if b.origin == o]
        if not sub:
            continue
        out += [f"### Occupied final cell, origin {o}", "",
                f"{len(sub)} children, {sum(b.accepted for b in sub)} accepted.", "",
                md_table(header, race_rows(sub)), ""]
    nacc = sum(b.accepted for b in empty)
    out += ["### Empty final cell", "",
            f"{len(empty)} children ({nacc} accepted, {pct(nacc / len(empty) if empty else float('nan'))}); "
            "there is no incumbent to race against, so none can be stopped. Their evaluations: "
            f"{sum(b.n for b in empty)} of {sum(b.n for b in bodies if b.z is not None)}."]
    return out


# ---------------------------------------------------------------- L6

def designs(b):
    """Reduced designs of one body: name -> fitness array [evals]."""
    z = b.z
    ts = np.asarray(z["task_scores"], float)           # [E, T, V]
    E, T, V = ts.shape
    C = z["metrics"].shape[3]
    full_t = ts.mean(axis=2)                           # [E, T]
    d = {"first variant": aggregate_cols(ts[:, :, 0])}
    if V > 1 or C > 1:
        alt = np.zeros((E, T))
        for e in range(E):
            for t in range(T):
                ms = [b.m(e, t, (t + c) % V, c) for c in range(C)]
                alt[e, t] = fit.task_score(ms, str(z["task"]))
        d["alternating variants"] = aggregate_cols(alt)
    if C > 1 and str(z["task"]) == "steer":
        st = np.zeros((E, T))
        for e in range(E):
            for t in range(T):
                st[e, t] = np.mean([fit.score(b.m(e, t, v, 0), "forward") for v in range(V)])
        d["straight command only"] = aggregate_cols(st)
    for t in range(T):
        d[f"terrain {z['terrains'][t]}"] = full_t[:, t]
    for t in range(T):
        for u in range(t + 1, T):
            d[f"terrains {z['terrains'][t]}+{z['terrains'][u]}"] = aggregate_cols(full_t[:, [t, u]])
    return d, aggregate_cols(full_t)


def section_l6(bodies):
    out = ["## L6: reduced evaluation designs", "",
           "Per CMA-ES generation the candidates are ranked by the full fitness (recomputed from "
           "the metrics) and by the reduced design. `spearman` and `overlap` (share of the full "
           "best half found in the reduced best half, what CMA-ES uses) are means over "
           "generations; `best in top quarter` is how often the generation's best candidate "
           "stays in the reduced top quarter.", ""]
    acc = {}
    checked = 0
    for b in bodies:
        if b.z is None or b.n < 4:
            continue
        d, full = designs(b)
        if not np.allclose(full, b.fitness, atol=1e-3):
            checked += 1
        gen = np.asarray(b.z["gen"])
        for g in np.unique(gen[gen >= 0]):
            idx = np.where(gen == g)[0]
            if len(idx) < 4:
                continue
            h, q = len(idx) // 2, max(len(idx) // 4, 1)
            top_full = set(idx[np.argsort(-full[idx], kind="stable")[:h]])
            best = idx[int(np.argmax(full[idx]))]
            for name, red in d.items():
                r = acc.setdefault(name, [[], [], []])
                r[0].append(spearman(full[idx], red[idx]))
                top_red = set(idx[np.argsort(-red[idx], kind="stable")[:h]])
                r[1].append(len(top_full & top_red) / h)
                r[2].append(best in set(idx[np.argsort(-red[idx], kind="stable")[:q]]))
    rows = []
    for name, (sp, ov, bq) in acc.items():
        rows.append([name, len(sp), num(np.nanmean(sp)), pct(np.mean(ov)), pct(np.mean(bq))])
    out.append(md_table(["design", "generations", "spearman", "overlap best half",
                         "best in top quarter"], rows) if rows else "no data")
    if checked:
        out += ["", f"Note: the recomputed full fitness differs from the stored one in {checked} "
                    "bodies (score rules on in the run?)."]
    return out


# ---------------------------------------------------------------- L7

def gain_shares(b):
    f = b.fitness
    best = np.maximum.accumulate(f)
    gain = best[-1] - f[0]
    sh = {}
    for frac in (0.25, 0.5, 0.75):
        k = max(int(math.ceil(frac * len(f))), 1)
        sh[frac] = (best[k - 1] - f[0]) / gain if gain > 1e-9 else None
    last = int(np.where(np.diff(best) > 0)[0].max() + 1) if gain > 1e-9 else 0
    return gain, sh, (len(f) - 1 - last) / len(f)


def section_l7(bodies):
    out = ["## L7: budgets", "",
           "Share of the final gain (best minus the start controller) reached after 25/50/75% "
           "of the evaluations, and evaluations after the last improvement as a share of the "
           "budget. Bodies with no gain are counted apart.", ""]
    used = [b for b in bodies if b.z is not None and b.n > 3]

    def row(name, bs):
        g = [gain_shares(b) for b in bs]
        with_gain = [x for x in g if x[1][0.25] is not None]
        if not with_gain:
            return [name, len(bs), len(bs) - len(with_gain), "-", "-", "-", "-", "-"]
        m = lambda f: np.mean([x[1][f] for x in with_gain])
        return [name, len(bs), len(bs) - len(with_gain), pct(m(0.25)), pct(m(0.5)), pct(m(0.75)),
                pct(np.mean([x[2] for x in g])), num(np.mean([x[0] for x in with_gain]))]

    header = ["group", "bodies", "no gain", "gain@25%", "gain@50%", "gain@75%",
              "evals after last improvement", "mean gain"]
    rows = [row("all", used)]
    rows += [row(f"origin {o}", [b for b in used if b.origin == o]) for o in ORIGINS
             if any(b.origin == o for b in used)]
    for lo, hi, name in ((0, 0, "0 changed slots"), (1, 1, "1 changed slot"),
                         (2, 2, "2 changed slots"), (3, 99, "3+ changed slots")):
        bs = [b for b in used if b.rec.get("changed_slots") is not None
              and lo <= b.rec["changed_slots"] <= hi]
        if bs:
            rows.append(row(name, bs))
    l2 = np.array([b.rec["l2"] for b in used if b.rec.get("l2") is not None])
    if len(l2) >= 6:
        cuts = np.quantile(l2, [1 / 3, 2 / 3])
        for i, name in enumerate(("L2 small third", "L2 middle third", "L2 large third")):
            lo = -np.inf if i == 0 else cuts[i - 1]
            hi = np.inf if i == 2 else cuts[i]
            bs = [b for b in used if b.rec.get("l2") is not None and lo <= b.rec["l2"] <= hi]
            rows.append(row(name, bs))
    out.append(md_table(header, rows))
    return out


# ---------------------------------------------------------------- L2

def steer_parts(ms):
    """(straight, turn, half) of one terrain; mirrors fitness.task_score."""
    turns = []
    for m, (c, _) in zip(ms[1:3], fit.commands("steer")[1:3]):
        dist = math.hypot(m.dx, m.dy)
        turn = min(fit.TURN_SCALE * min(-c * m.yaw, fit.TURN_CAP), dist)
        s = fit.up(turn, m) - fit.ENERGY_W * m.energy / (m.mass * fit.G)
        turns.append(s - fit.FALL_PENALTY if m.fell else s)
    full, slow = max(fit.up(ms[0].progress, ms[0]), 0.0), max(fit.up(ms[3].progress, ms[3]), 0.0)
    half = (fit.SPEED_ASKED * full - abs(slow - fit.SPEED_ASKED * full)
            - fit.ENERGY_W * ms[3].energy / (ms[3].mass * fit.G))
    if ms[3].fell:
        half -= fit.FALL_PENALTY
    return fit.score(ms[0], "forward"), float(np.mean(turns)), half


def section_l2(bodies):
    out = ["## L2 potential: where the gain of the inner loop comes from", ""]
    used = [b for b in bodies if b.z is not None and b.n > 3 and str(b.z["task"]) == "steer"]
    if not used:
        return out + ["no steer-task bodies in the trace."]
    w = (0.4, 0.4, 0.2)
    comp, tot, bad = [], [], 0
    for b in used:
        z = b.z
        T, V = z["metrics"].shape[1:3]
        e0, e1 = 0, int(np.argmax(b.fitness))

        def parts(e):
            p = np.array([[steer_parts([b.m(e, t, v, c) for c in range(4)]) for v in range(V)]
                          for t in range(T)])           # [T, V, 3]
            return p.mean(axis=(0, 1)), p
        p0, a0 = parts(e0)
        p1, a1 = parts(e1)
        chk = fit.task_score([b.m(e1, 0, 0, c) for c in range(4)], "steer")
        if abs(sum(wi * x for wi, x in zip(w, a1[0, 0])) - chk) > 1e-3:
            bad += 1
        comp.append([wi * (x1 - x0) for wi, x0, x1 in zip(w, p0, p1)])
        tot.append(sum(comp[-1]))
    comp, tot = np.array(comp), np.array(tot)
    pos = tot > 1e-9
    rows = [["straight (0.4)", num(comp[:, 0].mean()), pct(comp[pos, 0].sum() / tot[pos].sum()) if pos.any() else "-"],
            ["turns (0.4)", num(comp[:, 1].mean()), pct(comp[pos, 1].sum() / tot[pos].sum()) if pos.any() else "-"],
            ["half speed (0.2)", num(comp[:, 2].mean()), pct(comp[pos, 2].sum() / tot[pos].sum()) if pos.any() else "-"]]
    out += [f"{len(used)} bodies; gain = weighted component of the best candidate minus that of "
            "the start controller, mean over terrains and variants. Share = over bodies with a "
            "positive total gain.", "",
            md_table(["component", "mean gain per body", "share of the total gain"], rows), "",
            f"Bodies whose straight component alone is at least 80% of the total gain: "
            f"{pct(np.mean(comp[pos, 0] >= 0.8 * tot[pos])) if pos.any() else '-'}."]
    if bad:
        out.append(f"Check: the component formulas disagree with task_score in {bad} bodies.")
    return out


# ---------------------------------------------------------------- main

def report(directory):
    bodies = load(directory)
    head = [f"# Trace report: {directory}", "",
            f"{len(bodies)} bodies, {sum(b.rec['evals'] for b in bodies)} evaluations.", ""]
    parts = [head, section_general(bodies), section_l1(bodies), section_l6(bodies),
             section_l7(bodies), section_l2(bodies)]
    return "\n".join("\n".join(p) + "\n" for p in parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    text = report(a.dir)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        print(f"wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

"""
experiments/critic.py
---------------------
Numeric defect report for chosen robots of an archive, so a person reviews only
the robots without serious defects. A screening aid, never a score.

    python experiments/critic.py --archive A.pkl [--types a,b | --top N] \
        [--json out.json] [--workers N]

Every robot is judged as it stands: nominal flat ground, the archive's seconds,
payload, task and controller, with no re-optimisation.
"""

import argparse
import copy
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from darwinbot import morphology as mo   # noqa: E402
from darwinbot.sim import Episode        # noqa: E402
from darwinbot.terrain import get_terrains  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import prune                             # noqa: E402
from replay import open_archive          # noqa: E402

# Thresholds of the report; none of them enters a score.
USELESS_DROP = 0.05   # score share a cut may lose before the limb counts as useless
TURN_RATIO   = 0.5    # smaller / larger |yaw| of the two turn commands
DRAG_SHARE   = 0.3    # body or servo ground share
UPRIGHT_MIN  = 0.8
STILL_M      = 0.05   # m of straight progress
LIMB_SERVOS  = 4      # servos in one limb above which the limb is "heavy" (excess parts)
SEED = 0
DEFECTS = ("useless", "heavy", "one_side", "drags", "unstable", "still")
FIELDS = ("type", "servos", "nodes", "fitness", "progress", "useless",
          "useless_limbs", "limb_servos", "turn_ratio", "body_ground", "servo_ground",
          "tilt", "slip", "upright", "defects")

_ARCHIVE = None


def limbs(m):
    """Roots of the subtrees hanging from the core or a spine segment."""
    kids = {}
    for n in m.nodes.values():
        kids.setdefault(n.parent, []).append(n.id)
    return [c for i, n in m.nodes.items() if n.kind in ("core", "segment")
            for c in kids.get(i, []) if m.nodes[c].kind != "segment"]


def _servos(m, root):
    """Servos of one subtree (joints and wheels)."""
    return sum(mo.is_joint(m.nodes[i].kind) or m.nodes[i].kind == "wheel"
               for i in mo._subtree_ids(m, root))


def report_one(name, elite, a, seed=SEED):
    """The defect row of one elite, from its own morphology and controller."""
    m = mo.decode(elite.morph)
    flat = get_terrains("flat")[0]
    ep = Episode(m, flat, a.seconds, a.payload)
    straight = ep.run(elite.ctrl)
    base = prune._score([ep], elite.ctrl, a, seed)[0]  # archive's task scoring

    useless = []
    for root in limbs(m):
        trial = copy.deepcopy(m)            # as in prune.py: cut, keep a valid body
        mo._drop_subtree(trial, root)
        if not trial.n_servos:
            continue
        cut_ep = Episode(trial, flat, a.seconds, a.payload)
        if not cut_ep.buildable:
            continue
        cut = prune._score([cut_ep], elite.ctrl, a, seed)[0]
        drop = (base - cut) / max(abs(base), 1e-6)
        if drop < USELESS_DROP:
            useless.append({"slot": root, "kind": m.nodes[root].kind,
                            "servos": _servos(m, root), "drop": float(drop)})

    left = ep.run(elite.ctrl, None, -1.0, 1.0).yaw
    right = ep.run(elite.ctrl, None, 1.0, 1.0).yaw
    big = max(abs(left), abs(right))
    ratio = min(abs(left), abs(right)) / big if big > 1e-6 else 0.0

    limb_servos = sorted((_servos(m, r) for r in limbs(m)), reverse=True)
    defects = []
    if useless:
        defects.append("useless")
    if limb_servos and limb_servos[0] > LIMB_SERVOS:
        defects.append("heavy")
    if ratio < TURN_RATIO or left < 0 or right > 0:   # left turns left, right turns right
        defects.append("one_side")
    if straight.body_ground > DRAG_SHARE or straight.servo_ground > DRAG_SHARE:
        defects.append("drags")
    if straight.upright < UPRIGHT_MIN:
        defects.append("unstable")
    if straight.progress < STILL_M:
        defects.append("still")
    return {"type": name, "servos": m.n_servos, "nodes": m.n_nodes,
            "fitness": float(base), "progress": float(straight.progress),
            "useless": len(useless), "useless_limbs": useless,
            "limb_servos": limb_servos,
            "turn_ratio": float(ratio), "body_ground": float(straight.body_ground),
            "servo_ground": float(straight.servo_ground),
            "tilt": float(straight.tilt), "slip": float(straight.slip),
            "upright": float(straight.upright), "defects": defects}


def counts(rows):
    """Number of robots with each defect."""
    return {d: sum(d in r["defects"] for r in rows) for d in DEFECTS}


def _init(a):
    global _ARCHIVE
    _ARCHIVE = a
    a.activate()


def _job(job):
    name, elite, seed = job
    return report_one(name, elite, _ARCHIVE, seed)


def critic(a, names=None, top=None, workers=1, seed=SEED):
    """(rows, defect counts): the chosen robots of `a`, in the order given.
    Names come from Archive.find; --top takes the best elite per cell."""
    cells, labels = a.cells, a.names()
    if names:
        jobs = [(n, cells[a.find(n)], seed) for n in names]
    elif top:
        jobs = [(labels[c], e, seed) for c, e in
                sorted(cells.items(), key=lambda kv: -kv[1].fitness)[:top]]
    else:
        jobs = [(labels[c], e, seed) for c, e in cells.items()]
    if workers > 1:
        with Pool(workers, _init, (a,)) as pool:
            rows = pool.map(_job, jobs, chunksize=1)
    else:
        _init(a)
        rows = [_job(j) for j in jobs]
    return rows, counts(rows)


def show(rows, cnt):
    """Print one row per robot and the final per-defect counts."""
    print(f"{'type':26s} {'serv':>4s} {'nodes':>5s} {'fitness':>8s} {'prog':>7s} "
          f"{'use':>3s} {'limbs':>12s} {'ratio':>6s} {'body':>5s} {'servo':>5s} {'up':>5s}  defects")
    for r in rows:
        print(f"{r['type']:26s} {r['servos']:4d} {r['nodes']:5d} {r['fitness']:8.3f} "
              f"{r['progress']:7.3f} {r['useless']:3d} {'-'.join(map(str, r['limb_servos'])):>12s} {r['turn_ratio']:6.3f} "
              f"{r['body_ground']:5.2f} {r['servo_ground']:5.2f} {r['upright']:5.2f}  "
              f"{','.join(r['defects'])}")
    print("defects: " + ", ".join(f"{d} {cnt[d]}" for d in DEFECTS))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True)
    p.add_argument("--types", default=None, help="comma-separated robot names (default: all)")
    p.add_argument("--top", type=int, default=None, help="best N robots (by fitness)")
    p.add_argument("--json", default=None, metavar="OUT.json", help="write the same data as JSON")
    p.add_argument("--workers", type=int, default=1, help="worker processes (default: 1)")
    args = p.parse_args()
    if args.types and args.top:
        raise SystemExit("--types and --top are exclusive")
    a = open_archive(args.archive)
    rows, cnt = critic(a, args.types.split(",") if args.types else None,
                       args.top, args.workers)
    show(rows, cnt)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "counts": cnt}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

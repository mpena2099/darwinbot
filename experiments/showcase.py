"""
experiments/showcase.py
-----------------------
Take chosen robots of an archive through the whole showcase chain in one
command: polish, prune, validate, check the printed assembly, and snapshot.

    python experiments/showcase.py --archive merged.pkl --types a,b --out showcase --workers 8
    python experiments/showcase.py --archive a.pkl --types a --out s --skip polish,prune
    python experiments/showcase.py --archive a.pkl --types a,b --out s --strict-body   # drop belly robots

The output directory must not exist yet. It receives polished.pkl, pruned.pkl,
validate.txt, assembly.txt, showcase.snap.pkl and summary.txt (each robot's
score before and after polish and prune, its validate verdict and its clash
count). --skip leaves stages out, --budget is polish's CMA-ES budget.
--max-body SHARE (default evolve.BODY_SHARE) warns, before the chain, about each
robot whose stored body-ground share is above it, and validate fails it under the
same limit; --strict-body drops those robots from the chain instead.
"""

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from darwinbot.evolve import BODY_SHARE   # noqa: E402

STAGES = ("polish", "prune", "validate", "assembly", "snapshot")
FILES = {"polish": "polished.pkl", "prune": "pruned.pkl",
         "validate": "validate.txt", "assembly": "assembly.txt",
         "snapshot": "showcase.snap.pkl"}
PLUGIN_STAGES = ("polish", "prune", "validate", "snapshot")


def plan(skip=()):
    """Stage names to run, in chain order."""
    return [s for s in STAGES if s not in skip]


def command(stage, archive, types, out, workers=1, budget=1200, plugin=None,
            max_body=None):
    """argv of one stage, sys.executable first. Pure: nothing is read or run."""
    argv = _stage_argv(stage, archive, types, out, workers, budget)
    if plugin and stage in PLUGIN_STAGES:
        argv += ["--plugin", plugin]
    if max_body is not None and stage == "validate":
        argv += ["--max-body", str(max_body)]
    return argv


def _stage_argv(stage, archive, types, out, workers, budget):
    py = sys.executable
    paths = {k: os.path.join(out, v) for k, v in FILES.items()}
    if stage == "polish":
        return [py, os.path.join(ROOT, "experiments/polish.py"), "--archive", archive,
                "--types", ",".join(types), "--out", paths["polish"],
                "--budget", str(budget), "--workers", str(workers)]
    if stage == "prune":
        return [py, os.path.join(ROOT, "experiments/prune.py"), "--archive", archive,
                "--types", ",".join(types), "--out", paths["prune"]]
    if stage == "validate":
        return [py, os.path.join(ROOT, "experiments/validate.py"), "--archive", archive,
                "--type", types[0], "--workers", str(workers)]
    if stage == "assembly":
        return [py, os.path.join(ROOT, "tools/check_assembly.py"), "--archive", archive,
                "--type", types[0]]
    if stage == "snapshot":
        return [py, os.path.join(ROOT, "experiments/snapshot.py"), "--archive", archive,
                "--types", ",".join(types), "--out", paths["snapshot"]]
    raise ValueError(stage)


def _load(path):
    from darwinbot.evolve import load_archive
    return load_archive(path)


def _stored(path, types):
    """(fitness, body-ground share) of each named robot, in the given order."""
    a = _load(path)
    try:
        cells = [a.cells[a.find(t)] for t in types]
    except ValueError as e:
        raise SystemExit(f"{e}; see the types with replay.py --list")
    return [c.fitness for c in cells], [c.behaviour.body_ground for c in cells]


def _ordered(path, expected):
    """(fitness, name) of a stage archive, in insertion order (the --types order)."""
    a = _load(path)
    names = a.names()
    rows = [(a.cells[c].fitness, names[c]) for c in a.cells]
    if len(rows) != expected:
        raise SystemExit(f"{path} holds {len(rows)} robot(s), expected {expected}: "
                         "two robots landed in one cell; run that stage by hand")
    return rows


def _run(stage, argv, log=None):
    """Run one stage; with `log`, capture its output there and accept exit 1."""
    print(f"[{stage}] " + " ".join(argv[1:]), flush=True)
    r = subprocess.run(argv, capture_output=log is not None, text=True)
    text = ""
    if log is not None:
        text = (r.stdout or "") + (r.stderr or "")
        with open(log, "a") as f:
            f.write(f"=== {stage} ===\n{text}\n")
        print(text, end="", flush=True)
    if r.returncode not in ((0, 1) if log else (0,)) or "Traceback" in text:
        raise SystemExit(f"{stage} failed (exit {r.returncode})")
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description="darwinbot: the showcase chain in one command")
    p.add_argument("--archive", required=True)
    p.add_argument("--types", required=True,
                   help="comma-separated robot types, as replay.py --list prints them")
    p.add_argument("--out", required=True, help="new directory for the chain's files")
    p.add_argument("--workers", type=int, default=1, help="workers for polish and validate")
    p.add_argument("--budget", type=int, default=1200,
                   help="polish's CMA-ES evaluations per robot")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains the archive was trained with")
    p.add_argument("--skip", default="",
                   help="comma list of stages to leave out: " + ", ".join(STAGES))
    p.add_argument("--max-body", type=float, default=BODY_SHARE, metavar="SHARE",
                   help="warn about, and validate fail, a robot whose body is on the ground "
                        "more than this share of the time (default: %(default)s)")
    p.add_argument("--strict-body", action="store_true",
                   help="drop the robots over --max-body from the chain instead of warning")
    a = p.parse_args(argv)

    skip = [s.strip() for s in a.skip.split(",") if s.strip()]
    bad = [s for s in skip if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {list(STAGES)}")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; pick a new directory")
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    if not types:
        raise SystemExit("--types is empty: name at least one robot")
    run = plan(skip)

    fit_before, bodies = _stored(a.archive, types)
    slid = {t for t, b in zip(types, bodies) if b > a.max_body}
    for t, b in zip(types, bodies):
        if t in slid:
            print(f"warning: {t}: slides on its body {b:.0%} of the time", flush=True)
    if a.strict_body and slid:
        keep = [i for i, t in enumerate(types) if t not in slid]
        if not keep:
            raise SystemExit("every requested robot slides on its body; nothing to run")
        types = [types[i] for i in keep]
        fit_before = [fit_before[i] for i in keep]
    os.makedirs(a.out)
    fit_polish = fit_prune = fit_before
    names, verdicts = list(types), ["-"] * len(types)
    clashes = ["-"] * len(types)
    src = a.archive
    for stage in run:
        if stage in ("polish", "prune"):
            _run(stage, command(stage, src, names if stage == "prune" else types,
                                a.out, a.workers, a.budget, a.plugin))
            src = os.path.join(a.out, FILES[stage])
            rows = _ordered(src, len(types))
            names = [n for _, n in rows]
            if stage == "polish":
                fit_polish = [f for f, _ in rows]
            fit_prune = [f for f, _ in rows]
        elif stage in ("validate", "assembly"):
            for i, name in enumerate(names):
                text = _run(stage, command(stage, src, [name], a.out, a.workers,
                                           plugin=a.plugin, max_body=a.max_body),
                            os.path.join(a.out, FILES[stage]))
                if stage == "validate":
                    m = re.findall(r"\): (PASS|FAIL)", text)
                    verdicts[i] = m[-1] if m else "?"
                else:
                    m = re.search(r"(\d+) clash", text)
                    clashes[i] = int(m.group(1)) if m else "?"
        else:
            _run(stage, command(stage, src, names, a.out, plugin=a.plugin))

    lines = [f"{'type':<24} {'before':>8} {'polished':>9} {'pruned':>8} "
             f"{'validate':>9} {'clashes':>8}"]
    for row in zip(types, fit_before, fit_polish, fit_prune, verdicts, clashes):
        t, before, polished, pruned, verdict, clash = row
        lines.append(f"{t:<24} {before:8.3f} {polished:9.3f} {pruned:8.3f} "
                     f"{str(verdict):>9} {str(clash):>8}")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()

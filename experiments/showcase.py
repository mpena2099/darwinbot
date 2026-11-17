"""
experiments/showcase.py
-----------------------
Take chosen robots of an archive through the whole showcase chain in one
command: polish, prune, validate, check the printed assembly, snapshot and
export. Each robot is followed by its uid, so it keeps its identity when prune
changes its type name.

    python experiments/showcase.py --archive merged.pkl --types a,b --out showcase --workers 8
    python experiments/showcase.py --archive a.pkl --types a --out s --skip polish,prune
    python experiments/showcase.py --archive a.pkl --types a,b --out s --strict-body   # drop belly robots

The output directory must not exist yet. It receives polished.pkl, pruned.pkl,
validate.txt, final.pkl, assembly.txt, showcase.snap.pkl, exports/<name>/ and
summary.txt. Validate runs on the polished and on the pruned version of every
robot; the pruned one is chosen if it passes, else the polished one if it
passes, else the pruned one (marked FAIL). final.pkl holds the chosen version
of each robot, and assembly, snapshot and export read it. The summary lists
each robot's uid, name, score before and after polish and prune, both
verdicts, the chosen version and its clash count.
--skip leaves stages out, --budget is polish's CMA-ES budget.
--max-body SHARE (default evolve.BODY_SHARE) warns, before the chain, about each
robot whose stored body-ground share is above it, and validate fails it under the
same limit; --strict-body drops those robots from the chain instead.
"""

import argparse
import os
import pickle
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from darwinbot import evolve   # noqa: E402
from darwinbot.evolve import BODY_SHARE   # noqa: E402

STAGES = ("polish", "prune", "validate", "assembly", "snapshot", "export")
FILES = {"polish": "polished.pkl", "prune": "pruned.pkl",
         "validate": "validate.txt", "assembly": "assembly.txt",
         "snapshot": "showcase.snap.pkl", "export": "exports", "final": "final.pkl"}
PLUGIN_STAGES = ("polish", "prune", "validate", "snapshot", "export")


def choose(polished, pruned):
    """Version to ship from the two validate verdicts: the pruned one if it
    passes, else the polished one if it passes, else the pruned one."""
    if pruned == "PASS" or polished != "PASS":
        return "pruned"
    return "polished"


def plan(skip=()):
    """Stage names to run, in chain order."""
    return [s for s in STAGES if s not in skip]


def command(stage, archive, types, out, workers=1, budget=1200, plugin=None,
            max_body=None, name=None):
    """argv of one stage, sys.executable first. Pure: nothing is read or run.
    `types` are uids (or any name Archive.find takes); export writes to
    <out>/exports/<name>."""
    argv = _stage_argv(stage, archive, types, out, workers, budget, name)
    if plugin and stage in PLUGIN_STAGES:
        argv += ["--plugin", plugin]
    if max_body is not None and stage == "validate":
        argv += ["--max-body", str(max_body)]
    return argv


def _stage_argv(stage, archive, types, out, workers, budget, name=None):
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
    if stage == "export":
        return [py, os.path.join(ROOT, "experiments/export.py"), "--archive", archive,
                "--type", types[0],
                "--out", os.path.join(out, paths["export"], name or types[0])]
    raise ValueError(stage)


def _load(path):
    from darwinbot.evolve import load_archive
    return load_archive(path)


def _stored(path, types):
    """(uids, fitness, body-ground share) of each named robot, in the given order."""
    a = _load(path)
    try:
        cells = [a.cells[a.find(t)] for t in types]
    except ValueError as e:
        raise SystemExit(f"{e}; see the types with replay.py --list")
    return ([c.uid for c in cells], [c.fitness for c in cells],
            [c.behaviour.body_ground for c in cells])


def _by_uid(path, uids):
    """{uid: (fitness, name, elite)} of a stage archive for the followed robots."""
    a = _load(path)
    names = a.names()
    rows = {e.uid: (e.fitness, names[c], e) for c, e in a.cells.items()}
    missing = [u for u in uids if u not in rows]
    if missing:
        raise SystemExit(f"{path} lacks robot(s) {missing}: two robots landed in one "
                         "cell; run that stage by hand")
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

    uids, fit_before, bodies = _stored(a.archive, types)
    slid = {u for u, b in zip(uids, bodies) if b > a.max_body}
    for t, u, b in zip(types, uids, bodies):
        if u in slid:
            print(f"warning: {t}: slides on its body {b:.0%} of the time", flush=True)
    if a.strict_body and slid:
        keep = [i for i, u in enumerate(uids) if u not in slid]
        if not keep:
            raise SystemExit("every requested robot slides on its body; nothing to run")
        uids = [uids[i] for i in keep]
        fit_before = [fit_before[i] for i in keep]
    ids = [str(u) for u in uids]
    os.makedirs(a.out)
    path = {k: os.path.join(a.out, v) for k, v in FILES.items()}
    n = len(uids)
    polished = pruned = a.archive           # the archive of each version
    fit_polish = fit_prune = fit_before
    names = _names(a.archive, uids)
    final_names = names
    verdict = {"polished": ["-"] * n, "pruned": ["-"] * n}
    chosen, clashes = ["pruned"] * n, ["-"] * n
    final = None

    def validate_all():
        seen = {}
        for version, src in (("polished", polished), ("pruned", pruned)):
            if src in seen:                 # prune skipped: one archive, one run
                verdict[version] = verdict[seen[src]]
                continue
            seen[src] = version
            for i, uid in enumerate(ids):
                text = _run(f"validate {version} {uid}",
                            command("validate", src, [uid], a.out, a.workers,
                                    plugin=a.plugin, max_body=a.max_body),
                            path["validate"])
                m = re.findall(r"\): (PASS|FAIL)", text)
                verdict[version][i] = m[-1] if m else "?"
        for i in range(n):
            chosen[i] = choose(verdict["polished"][i], verdict["pruned"][i])

    def write_final():
        nonlocal final, final_names
        rows = {"polished": _by_uid(polished, uids), "pruned": _by_uid(pruned, uids)}
        out = _load(pruned).like()
        for i, uid in enumerate(uids):
            if not out.insert(rows[chosen[i]][uid][2]):
                raise SystemExit(f"{names[i]} and another chosen robot share a cell, "
                                 "so one final.pkl cannot hold both; use polished.pkl "
                                 "and pruned.pkl by hand")
        with open(path["final"], "wb") as f:
            pickle.dump(out, f)
        final = path["final"]
        final_names = [out.names()[evolve.cell_of(rows[chosen[i]][uid][2].behaviour,
                                                  out.n_descriptors)]
                       for i, uid in enumerate(uids)]

    for stage in run:
        if stage in ("polish", "prune"):
            src = polished if stage == "prune" else a.archive
            _run(stage, command(stage, src, ids, a.out, a.workers, a.budget, a.plugin))
            if stage == "polish":
                polished = pruned = path["polish"]
                rows = _by_uid(polished, uids)
                fit_polish = fit_prune = [rows[u][0] for u in uids]
            else:
                pruned = path["prune"]
                rows = _by_uid(pruned, uids)
                fit_prune = [rows[u][0] for u in uids]
            names = [rows[u][1] for u in uids]
        elif stage == "validate":
            validate_all()
        else:
            if final is None:
                write_final()
            for i, uid in enumerate(ids):
                if stage == "assembly":
                    text = _run(stage, command(stage, final, [uid], a.out, a.workers,
                                               plugin=a.plugin), path["assembly"])
                    m = re.search(r"(\d+) clash", text)
                    clashes[i] = int(m.group(1)) if m else "?"
                elif stage == "export":
                    _run(stage, command(stage, final, [uid], a.out, plugin=a.plugin,
                                        name=final_names[i]))
            if stage == "snapshot":
                _run(stage, command(stage, final, ids, a.out, plugin=a.plugin))
    if final is None:
        write_final()

    lines = [f"{'uid':>5} {'name':<26} {'before':>8} {'polished':>9} {'pruned':>8} "
             f"{'val.pol':>8} {'val.pru':>8} {'chosen':>9} {'clashes':>8}"]
    for i, uid in enumerate(uids):
        lines.append(f"{uid:>5} {final_names[i]:<26} {fit_before[i]:8.3f} "
                     f"{fit_polish[i]:9.3f} {fit_prune[i]:8.3f} "
                     f"{verdict['polished'][i]:>8} {verdict['pruned'][i]:>8} "
                     f"{chosen[i]:>9} {str(clashes[i]):>8}")
    summary = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary, end="")


def _names(path, uids):
    rows = _by_uid(path, uids)
    return [rows[u][1] for u in uids]


if __name__ == "__main__":
    main()

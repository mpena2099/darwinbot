"""
Bring an archive from an older BODY_MODEL up to the current one, so a bump does
not throw a trained run away: the genes are translated (evolve.MIGRATIONS),
every elite is checked and its controller is searched again under the current
code, and the result is a new archive to resume from.

    python experiments/migrate_archive.py old.pkl --out new.pkl [--budget 60]

The starting controller of each elite is kept when the search finds nothing
better. A body that is no longer buildable is dropped. Terrains are the fixed
layouts, as in rescore_archive.py. An archive trained on the default hardware
takes the current default profile (otherwise a change to it, such as the servo
mass, would never apply); a custom kit is kept (--profile keep) or replaced
(--profile default).
"""

import argparse
import os
import pickle
import sys
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np   # noqa: E402

from darwinbot import evolve, hardware, morphology as morpho, plugins   # noqa: E402
from darwinbot.evolve import Archive, Elite, elite_id   # noqa: E402
from darwinbot.inner import optimise_controller   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402


def is_default_profile(profile: dict) -> bool:
    """Whether a stored profile is the default kit of its time: every value it
    shares with today's default is equal (keys one side lacks are what a
    hardware change adds or renames)."""
    current = hardware.default_profile()
    profile = hardware.normalize(profile)
    for section, values in profile.items():
        for key, value in values.items():
            if key in current.get(section, {}) and current[section][key] != value:
                return False
    return True


def choose_profile(profile: dict, mode: str) -> tuple:
    """(profile to migrate onto, message or None). mode: auto, default, keep."""
    current = hardware.default_profile()
    if mode == "default" or (mode == "auto" and is_default_profile(profile)):
        return current, None
    if mode == "auto":
        return profile, ("the archive's hardware profile is not the default kit: "
                         "keeping it (--profile default replaces it)")
    return profile, None


def check_keys(profile: dict) -> None:
    """A kept profile must have the keys this code reads; hardware.require would
    otherwise fail on import after the re-run."""
    current = hardware.default_profile()
    profile = hardware.normalize(profile)
    for section, values in current.items():
        wrong = set(values) ^ set(profile.get(section, {}))
        if wrong:
            raise SystemExit(
                f"the archive's hardware profile does not fit this code ('{section}.{sorted(wrong)[0]}'): "
                "migrate it with --profile default, or edit the kit and seed a new archive")


def _migrate_one(job):
    genes, ctrl, steps, terrains, seconds, variants, task, payload, budget, seed = job
    try:
        for step in steps:
            genes, ctrl = evolve.MIGRATIONS[step](genes, ctrl)
        morph = morpho.decode(genes)
        res = optimise_controller(
            morph, get_terrains(terrains), x0=ctrl,
            budget=evolve.scaled_budget(budget, morph), sigma0=0.12, seed=seed,
            seconds=seconds, variants=variants, task=task, payload=payload)
    except Exception:
        print(f"body failed:\n{traceback.format_exc()}", flush=True)
        return genes, ctrl, None, True
    return genes, res.ctrl, res, False


# Body models 20 and 21 used a 4-axis, 500-cell grid; cell ids compare only from 22 on.
SAME_GRID_SINCE = 22


def migrate(archive: Archive, workers: int = 1, budget: int = 60, seconds: float = None) -> tuple:
    """(new archive at the current BODY_MODEL, summary dict). `archive` must be
    active (archive.activate()) and loaded with evolve.loading_stale()."""
    version = archive.body_model
    steps = evolve.migration_path(version)
    if steps is None:
        raise ValueError(f"no migration from body model {version} to {evolve.BODY_MODEL}")
    old = sorted(archive.cells.items())
    seconds_before = archive.seconds
    if seconds is not None:
        archive.seconds = seconds
    jobs = []
    for _, e in old:
        seed = int(elite_id(e.morph, e.ctrl), 16) % 2**31       # fixed per elite
        jobs.append((e.morph, e.ctrl, steps, archive.terrains, archive.seconds,
                     archive.variants, archive.task, archive.payload, budget, seed))
    if workers > 1:
        with Pool(workers, evolve._init_worker, (archive.ablate, archive.rules,
                                                 archive.body, plugins.loaded())) as pool:
            results = pool.map(_migrate_one, jobs)
    else:
        results = [_migrate_one(j) for j in jobs]

    out = archive.like()
    out.seconds = archive.seconds
    out.lineage = dict(archive.lineage)
    dropped = failed = changed = merged = 0
    same_grid = version >= SAME_GRID_SINCE
    before = [e.fitness for _, e in old]
    for (cell, e), (genes, ctrl, res, bad) in zip(old, results):
        failed += bad
        if res is None or res.evals == 0:      # no longer buildable
            dropped += 1
            continue
        old_id = elite_id(e.morph, e.ctrl)
        new = Elite(genes, res.ctrl, res.fitness, res.scores, e.born, res.behaviour)
        if not out.insert(new, old_id):        # a better elite holds the cell
            merged += 1
        else:
            uid = elite_id(new.morph, new.ctrl)
            if uid == old_id and old_id in archive.lineage:
                out.lineage[uid] = (archive.lineage[old_id][0],) + out.lineage[uid][1:]
        if same_grid:
            changed += evolve.cell_of(res.behaviour) != cell
    if not out.cells:
        raise ValueError("no elite survived the migration")
    after = [e.fitness for e in out.cells.values()]
    summary = {"from": version, "elites": len(old), "dropped": dropped,
               "merged": merged, "failed": failed, "seconds_before": seconds_before,
               "seconds_after": archive.seconds, "cells": len(after), "changed_cell": int(changed) if same_grid else None,
               "mean_before": float(np.mean(before)), "mean_after": float(np.mean(after)),
               "best_before": max(before), "best_after": max(after)}
    s = out.stats()
    s.update(added=0, seconds=0.0, migrated_from=version, cells_before=len(old),
             cells_after=len(after), dropped=dropped,
             mean_before=summary["mean_before"], mean_after=summary["mean_after"])
    out.history = list(archive.history) + [s]
    return out, summary


def print_summary(s: dict) -> None:
    print(f"migrated from body model {s['from']} to {evolve.BODY_MODEL}")
    print(f"episode length: {s['seconds_before']} s before, {s['seconds_after']} s after")
    print(f"{'elites in':>10} {'dropped':>8} {'merged':>7} {'cells out':>10} {'changed cell':>13} "
          f"{'mean before':>12} {'mean after':>11} {'best before':>12} {'best after':>11}")
    changed = "n/a" if s["changed_cell"] is None else s["changed_cell"]
    print(f"{s['elites']:>10} {s['dropped']:>8} {s['merged']:>7} {s['cells']:>10} {changed:>13} "
          f"{s['mean_before']:>12.2f} {s['mean_after']:>11.2f} "
          f"{s['best_before']:>12.2f} {s['best_after']:>11.2f}")
    if s["changed_cell"] is None:
        print("cell change: n/a (new grid)")
    if s["merged"]:
        print(f"{s['merged']} re-evaluated elites merged into an occupied cell")
    if s["failed"]:
        print(f"{s['failed']} of the dropped bodies failed with an error (see above)")


def main():
    p = argparse.ArgumentParser(description="darwinbot: migrate an archive to the current body model")
    p.add_argument("archive")
    p.add_argument("--out", required=True, help="new archive; never overwritten")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains the archive was trained with")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--budget", type=int, default=60,
                   help="CMA-ES evaluations per elite before the joint-count scaling")
    p.add_argument("--seconds", type=float, default=None,
                   help="episode length for the new evaluation, stored in the result "
                        "(default: the archive's own)")
    p.add_argument("--profile", choices=("auto", "default", "keep"), default="auto",
                   help="hardware profile of the result: auto takes the current default "
                        "when the archive was trained on the default kit and keeps a custom kit")
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; the migration never overwrites")
    archive = evolve.load_archive(a.archive, stale=True)
    if archive.body_model == evolve.BODY_MODEL:
        raise SystemExit("the archive is already at the current body model")
    if evolve.migration_path(archive.body_model) is None:
        raise SystemExit(f"no migration from body model {archive.body_model}: retrain")
    archive.profile, note = choose_profile(archive.profile, a.profile)
    if note:
        print(note)
    plugins.require(archive, a.plugin)
    check_keys(archive.profile)
    hardware.require(archive.profile)         # may re-run this command under that profile
    archive.activate()
    out, summary = migrate(archive, a.workers, a.budget, a.seconds)
    with open(a.out + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(a.out + ".tmp", a.out)
    print_summary(summary)


if __name__ == "__main__":
    main()

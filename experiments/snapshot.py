"""
experiments/snapshot.py
-----------------------
Record robots from archives into a code-independent snapshot: the compiled
MJCF and the torso poses of one flat-ground run, so a later BODY_MODEL can
still play them.

  python experiments/snapshot.py --archive a.pkl
  python experiments/snapshot.py --archive a.pkl --types 8servos-low-crawl --seconds 8
  python experiments/snapshot.py --robot a.pkl:3servos-low-roll-1:flat-roller --robot b.pkl:10servos-mid-belly-0 --out showcase.snap.pkl

--robot ARCHIVE:TYPE[:NAME] (repeatable) takes one robot from any archive and
stores it under NAME (default TYPE), so one snapshot can hold robots of
several releases. It replaces --archive and --types, needs --out, and the
archives must share body model, task and seconds.
"""

import argparse
import os
import pickle

import numpy as np

from darwinbot import hardware, plugins
from darwinbot import evolve
from darwinbot import morphology as morpho
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains

DEFAULT_ARCHIVE = "checkpoints/archive.pkl"


def load_archive(path, plugin=None):
    archive = evolve.load_archive(path)
    plugins.require(archive, plugin)
    hardware.require(archive.profile)
    archive.activate()
    return archive


def parse_robot(spec):
    """'ARCHIVE:TYPE' or 'ARCHIVE:TYPE:NAME' -> (archive, type, name)."""
    parts = [s.strip() for s in spec.split(":")]
    if len(parts) not in (2, 3) or not all(parts):
        raise SystemExit(f"--robot {spec!r}: give ARCHIVE:TYPE or ARCHIVE:TYPE:NAME")
    archive, kind = parts[0], parts[1]
    return archive, kind, (parts[2] if len(parts) == 3 else kind)


def check_robot_names(specs):
    names = [name for _, _, name in specs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"two robots would be named {dup}: give each one its own "
                         "NAME with ARCHIVE:TYPE:NAME")


def check_agree(loaded):
    """loaded: [(path, archive)]. One snapshot has one body model, task and seconds."""
    first_path, first = loaded[0]
    for path, other in loaded[1:]:
        for attr in ("body_model", "seconds", "task"):
            x, y = getattr(first, attr), getattr(other, attr)
            if x != y:
                raise SystemExit(
                    f"archives disagree on {attr}: {first_path} has {x!r}, "
                    f"{path} has {y!r}; a snapshot needs archives with one {attr}")


def record(episode, ctrl, fps):
    """Torso qpos after every 1/fps of simulated time, walking straight."""
    cpg = episode.reset(ctrl, command=0.0)
    dt = episode.model.opt.timestep
    every = max(1, int(round(1.0 / (fps * dt))))
    frames = []
    for i in range(1, episode.n_steps + 1):
        episode.step(cpg)
        if i % every == 0:
            frames.append(episode.data.qpos.copy())
    qpos = np.asarray(frames, dtype=np.float32).reshape(-1, episode.model.nq)
    return qpos, 1.0 / (every * dt)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="darwinbot: record archive robots into a code-independent snapshot")
    p.add_argument("--archive", default=None,
                   help=f"default: {DEFAULT_ARCHIVE}; not with --robot")
    p.add_argument("--robot", action="append", default=None, metavar="ARCHIVE:TYPE[:NAME]",
                   help="one robot from any archive, stored under NAME (default TYPE); "
                        "repeat for more robots")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains the archive was trained with")
    p.add_argument("--types", default=None,
                   help="comma-separated robot types from replay.py --list (default: all)")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None, metavar="OUT.pkl",
                   help="default: the archive path with .pkl replaced by .snap.pkl "
                        "(required with --robot)")
    a = p.parse_args(argv)

    if a.robot and (a.archive is not None or a.types):
        raise SystemExit("--robot replaces --archive and --types: give one --robot per robot")
    if a.robot and not a.out:
        raise SystemExit("--robot needs --out: the robots come from several archives")
    if a.robot:
        out = a.out
    else:
        archive_path = a.archive or DEFAULT_ARCHIVE
        out = a.out or (archive_path[:-4] + ".snap.pkl" if archive_path.endswith(".pkl")
                        else archive_path + ".snap.pkl")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: pick another --out")

    if a.robot:
        specs = [parse_robot(s) for s in a.robot]
        check_robot_names(specs)
        loaded = {}
        for path, _, _ in specs:
            if path not in loaded:
                loaded[path] = load_archive(path, a.plugin)
                if not loaded[path].cells:
                    raise SystemExit(f"{path} has no elites")
        check_agree(list(loaded.items()))
        jobs = []
        for path, kind, name in specs:
            try:
                cell = loaded[path].find(kind)
            except ValueError as e:
                raise SystemExit(f"{path}: {e}; see the available types with replay.py --list")
            jobs.append((loaded[path], cell, name))
        task_archive = loaded[specs[0][0]]
    else:
        archive = load_archive(archive_path, a.plugin)
        if not archive.cells:
            raise SystemExit(f"{archive_path} has no elites")
        names = archive.names()
        if a.types:
            cells = []
            for name in (t.strip() for t in a.types.split(",") if t.strip()):
                try:
                    cells.append(archive.find(name))
                except ValueError as e:
                    raise SystemExit(f"{e}; see the available types with replay.py --list")
        else:
            cells = sorted(names)
        jobs = [(archive, cell, names[cell]) for cell in cells]
        task_archive = archive

    terrain = get_terrains("flat")[0]
    robots = {}
    for archive, cell, name in jobs:
        archive.activate()                 # the robot was trained with these rules
        elite = archive.cells[cell]
        episode = Episode(morpho.decode(elite.morph), terrain,
                          payload=archive.payload, seconds=a.seconds)
        qpos, fps = record(episode, elite.ctrl, a.fps)
        robots[name] = {"xml": episode.xml, "fps": round(fps, 3), "qpos": qpos,
                        "fitness": float(elite.fitness)}
        print(f"{name}: {len(qpos)} frames, nq {qpos.shape[1]}")

    snap = {"format": 1, "body_model": evolve.BODY_MODEL, "task": task_archive.task,
            "seconds": a.seconds, "robots": robots}
    with open(out, "wb") as f:
        pickle.dump(snap, f)
    print(f"wrote {out} ({len(robots)} robots)")


if __name__ == "__main__":
    main()

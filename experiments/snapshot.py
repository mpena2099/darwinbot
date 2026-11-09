"""
experiments/snapshot.py
-----------------------
Record robots from an archive into a code-independent snapshot: the compiled
MJCF and the torso poses of one flat-ground run, so a later BODY_MODEL can
still play them.

  python experiments/snapshot.py --archive a.pkl
  python experiments/snapshot.py --archive a.pkl --types 8servos-low-crawl --seconds 8
"""

import argparse
import os
import pickle

import numpy as np

from darwinbot import ablate, hardware
from darwinbot import evolve
from darwinbot import morphology as morpho
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains


def load_archive(path):
    try:
        with open(path, "rb") as f:
            archive = pickle.load(f)
        hardware.require(archive.profile)
        return archive
    except FileNotFoundError:
        raise SystemExit(f"{path} not found: train first with experiments/train.py")


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
    return np.asarray(frames, dtype=np.float32).reshape(-1, episode.model.nq)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="darwinbot: record archive robots into a code-independent snapshot")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--types", default=None,
                   help="comma-separated robot types from replay.py --list (default: all)")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None, metavar="OUT.pkl",
                   help="default: the archive path with .pkl replaced by .snap.pkl")
    a = p.parse_args(argv)

    out = a.out or (a.archive[:-4] + ".snap.pkl" if a.archive.endswith(".pkl")
                    else a.archive + ".snap.pkl")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: pick another --out")

    archive = load_archive(a.archive)
    if not archive.cells:
        raise SystemExit(f"{a.archive} has no elites")
    ablate.set_active(archive.ablate)      # the robots were trained with these rules off
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

    terrain = get_terrains("flat")[0]
    robots = {}
    for cell in cells:
        elite = archive.cells[cell]
        name = names[cell]
        episode = Episode(morpho.decode(elite.morph), terrain,
                          payload=archive.payload, seconds=a.seconds)
        qpos = record(episode, elite.ctrl, a.fps)
        robots[name] = {"xml": episode.xml, "fps": a.fps, "qpos": qpos,
                        "fitness": float(elite.fitness)}
        print(f"{name}: {len(qpos)} frames, nq {qpos.shape[1]}")

    snap = {"format": 1, "body_model": evolve.BODY_MODEL, "task": archive.task,
            "seconds": a.seconds, "robots": robots}
    with open(out, "wb") as f:
        pickle.dump(snap, f)
    print(f"wrote {out} ({len(robots)} robots)")


if __name__ == "__main__":
    main()

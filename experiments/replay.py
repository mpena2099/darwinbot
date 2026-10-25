"""
experiments/replay.py
---------------------
Replay an evolved body and gait from an archive checkpoint.

  python experiments/replay.py                              # best elite, flat, viewer
  python experiments/replay.py --list                       # robot types found
  python experiments/replay.py --type 8servos-low-crawl --terrain rough
  python experiments/replay.py --record out.mp4 --seconds 8
"""

import argparse
import pickle
import signal
import time

import mujoco

from darwinbot import morphology as morpho
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains


def open_archive(path):
    """Unpickle an archive, turning the usual failures into a plain message."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found: train first with experiments/train.py")
    except ValueError as e:
        raise SystemExit(f"{path}: {e}. Train a new archive with experiments/train.py "
                         f"(or pass --archive with a newer one).")


def load(path, cell):
    archive = open_archive(path)
    if not archive.cells:
        raise SystemExit(f"{path} has no elites")
    if cell is None:
        return archive, archive.best()
    try:
        key = archive.find(cell)
    except ValueError as e:
        raise SystemExit(f"{e}; see the available types with --list")
    return archive, archive.cells[key]


def record(episode, ctrl, path, seconds, fps=30):
    try:
        import imageio
    except ImportError:
        print("imageio is not installed: pip install 'darwinbot[video]'")
        print("or: pip install imageio[ffmpeg]")
        return

    renderer = mujoco.Renderer(episode.model, height=480, width=640)
    cpg = episode.reset(ctrl)
    dt = episode.model.opt.timestep
    every = max(1, int(round(1.0 / (fps * dt))))
    fps = 1.0 / (every * dt)       # the rate the frame step really gives
    writer = imageio.get_writer(path, fps=fps)
    frames = 0
    try:
        for i in range(int(seconds / dt)):
            episode.step(cpg)
            if i % every == 0:
                renderer.update_scene(episode.data, camera="track")
                writer.append_data(renderer.render())
                frames += 1
    finally:
        writer.close()
        renderer.close()
    print(f"wrote {path} ({frames} frames, {frames / fps:.1f} s)")


def watch(episode, ctrl, speed=1.0):
    from mujoco import viewer

    cpg = episode.reset(ctrl)
    dt = episode.model.opt.timestep
    wall_dt = dt / speed     # real seconds per simulation step

    # Ctrl+C must not raise inside the viewer context: the exception unwinds
    # while the render thread is still drawing, the close hangs and repeated
    # Ctrl+C ends in a segfault. A flag lets the loop exit and close cleanly.
    stop = []
    previous = signal.signal(signal.SIGINT, lambda *_: stop.append(True))
    try:
        with viewer.launch_passive(episode.model, episode.data) as handle:
            while handle.is_running() and not stop:
                t0 = time.perf_counter()
                episode.step(cpg)
                handle.sync()
                time.sleep(max(0.0, wall_dt - (time.perf_counter() - t0)))
            handle.close()
        # close() returns before the render thread has released GLFW; exiting
        # the interpreter right away segfaults, so give the thread a moment.
        time.sleep(0.5)
    finally:
        signal.signal(signal.SIGINT, previous)
    if stop:
        print("stopped")


def main():
    p = argparse.ArgumentParser(description="darwinbot: replay an evolved robot")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--type", dest="cell", default=None,
                   help="robot type from --list, e.g. 8servos-low-crawl "
                        "(default: best robot overall)")
    p.add_argument("--terrain", default="flat")
    p.add_argument("--record", default=None, metavar="OUT.mp4",
                   help="render offscreen instead of opening the viewer")
    p.add_argument("--seconds", type=float, default=8.0,
                   help="recording length")
    p.add_argument("--speed", type=float, default=1.0,
                   help="viewer playback speed, e.g. 0.25 for slow motion")
    p.add_argument("--list", action="store_true",
                   help="print every archive cell with its fitness, duty, height and servos, then exit")
    a = p.parse_args()

    if a.list:
        archive = open_archive(a.archive)
        names = archive.names()
        if archive.task != "forward" or archive.payload:
            print(f"task: {archive.task}, payload {archive.payload:g} kg")
        print("type                       fitness  duty  height  servos    cot")
        for cell, e in sorted(archive.cells.items(), key=lambda kv: -kv[1].fitness):
            duty, height, servos, _, cot = e.behaviour
            print(f"{names[cell]:26s} {e.fitness:7.2f}  {duty:4.2f}  {height:6.2f}  "
                  f"{int(servos):6d}  {cot:5.1f}")
        return

    archive, elite = load(a.archive, a.cell)
    morph = morpho.decode(elite.morph)
    print(morpho.summary(morph))
    print(f"fitness {elite.fitness:.3f}  (task {archive.task}, payload {archive.payload:g} kg)")
    for terrain, s in zip(get_terrains(archive.terrains), elite.scores):
        print(f"  {terrain.name:9s} {s:+.3f}")

    terrain = get_terrains(a.terrain)[0]
    episode = Episode(morph, terrain, payload=archive.payload)
    print(f"terrain {terrain.name}")
    if a.record:
        record(episode, elite.ctrl, a.record, a.seconds)
    else:
        watch(episode, elite.ctrl, a.speed)


if __name__ == "__main__":
    main()

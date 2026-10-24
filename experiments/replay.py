"""
experiments/replay.py
---------------------
Replay an evolved body and gait from an archive checkpoint.

  python experiments/replay.py                              # best elite, flat, viewer
  python experiments/replay.py --list                       # robot types found
  python experiments/replay.py --type 6legs-2wheels-14servos --terrain rough
  python experiments/replay.py --record out.mp4 --seconds 8
"""

import argparse
import pickle
import signal
import time

import mujoco

from darwinbot import morphology as morpho
from darwinbot.evolve import cell_name, parse_cell_name
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains


def load(path, cell):
    with open(path, "rb") as f:
        archive = pickle.load(f)
    if not archive.cells:
        raise SystemExit(f"{path} has no elites")
    if cell is None:
        return archive, archive.best()
    try:
        key = parse_cell_name(cell)
    except ValueError as e:
        raise SystemExit(str(e))
    if key not in archive.cells:
        raise SystemExit(f"no {cell} robot in the archive yet; "
                         f"see the available types with --list")
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
                   help="robot type from --list, e.g. 6legs-2wheels-14servos "
                        "(default: best robot overall)")
    p.add_argument("--terrain", default="flat")
    p.add_argument("--record", default=None, metavar="OUT.mp4",
                   help="render offscreen instead of opening the viewer")
    p.add_argument("--seconds", type=float, default=8.0,
                   help="recording length")
    p.add_argument("--speed", type=float, default=1.0,
                   help="viewer playback speed, e.g. 0.25 for slow motion")
    p.add_argument("--list", action="store_true",
                   help="print every archive cell with its fitness and exit")
    a = p.parse_args()

    if a.list:
        with open(a.archive, "rb") as f:
            archive = pickle.load(f)
        print("type                    fitness  leg length  servos")
        for cell, e in sorted(archive.cells.items(), key=lambda kv: -kv[1].fitness):
            m = morpho.decode(e.morph)
            print(f"{cell_name(cell):22s} {e.fitness:8.2f}  {m.mean_leg_length:.2f} m     {m.n_servos}")
        return

    archive, elite = load(a.archive, a.cell)
    morph = morpho.decode(elite.morph)
    print(morpho.summary(morph))
    print(f"fitness {elite.fitness:.3f}")
    for terrain, s in zip(get_terrains(archive.terrains), elite.scores):
        print(f"  {terrain.name:9s} {s:+.3f}")

    terrain = get_terrains(a.terrain)[0]
    episode = Episode(morph, terrain)
    print(f"terrain {terrain.name}")
    if a.record:
        record(episode, elite.ctrl, a.record, a.seconds)
    else:
        watch(episode, elite.ctrl, a.speed)


if __name__ == "__main__":
    main()

"""
experiments/replay.py
---------------------
Replay an evolved body and gait from an archive checkpoint.

  python experiments/replay.py                              # best elite, flat, viewer
  python experiments/replay.py --list                       # robot types found
  python experiments/replay.py --list --where "wheels=0,legs>=4"   # only the matching ones
  python experiments/replay.py --type 8servos-low-crawl --terrain rough
  python experiments/replay.py --record out.mp4 --seconds 8
  python experiments/replay.py --robot my_robot.yaml         # a hand-written body, standing
"""

import argparse
import operator
import pickle
import re
import signal
import time

import mujoco
import numpy as np

from darwinbot import hardware, plugins, robot_file
from darwinbot import controller as ctl
from darwinbot.controller import Params
from darwinbot import morphology as morpho
from darwinbot.sim import Episode
from darwinbot.terrain import get_terrains


WHERE_FIELDS = ("fitness", "duty", "height", "servos", "legs", "cot",
                "wheels", "feet", "segments", "passive", "nodes")
WHERE_OPS = {"=": operator.eq, "!=": operator.ne, "<": operator.lt,
             "<=": operator.le, ">": operator.gt, ">=": operator.ge}
PASSIVE_KINDS = ("spring", "tail", "skid", "spike", "caster")


def parse_where(expr):
    """Turn 'field OP number, ...' into (field, op, value) triples. Every
    condition must hold; a bad one stops with a message naming the fields."""
    conds = []
    for part in expr.split(","):
        m = re.fullmatch(r"\s*(\w*)\s*(!=|<=|>=|=|<|>)\s*(.*?)\s*", part)
        if not m:
            raise SystemExit(f"--where: bad condition {part.strip()!r}, expected "
                             f"field OP number, e.g. legs>=4 (fields: "
                             f"{', '.join(WHERE_FIELDS)}; OP: {' '.join(WHERE_OPS)})")
        field, op, text = m.groups()
        if field not in WHERE_FIELDS:
            raise SystemExit(f"--where: unknown field {field!r} (fields: "
                             f"{', '.join(WHERE_FIELDS)})")
        try:
            value = float(text)
        except ValueError:
            raise SystemExit(f"--where: {text!r} in {part.strip()!r} is not a number")
        conds.append((field, op, value))
    return conds


def matches(row, conds):
    """True when the row (field name to value) meets every condition."""
    return all(WHERE_OPS[op](row[field], value) for field, op, value in conds)


def open_archive(path, plugin=None):
    """Unpickle an archive, turning the usual failures into a plain message, and
    turn on the ablations, score rules and body limits it was trained with."""
    try:
        with open(path, "rb") as f:
            archive = pickle.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found: train first with experiments/train.py")
    except ValueError as e:
        raise SystemExit(f"{path}: {e}. Train a new archive with experiments/train.py "
                         f"(or pass --archive with a newer one).")
    plugins.require(archive, plugin)
    hardware.require(archive.profile)    # re-runs the command under the archive's hardware
    archive.activate()
    return archive


def load(path, cell, plugin=None):
    archive = open_archive(path, plugin)
    if not archive.cells:
        raise SystemExit(f"{path} has no elites")
    if cell is None:
        return archive, archive.best()
    try:
        key = archive.find(cell)
    except ValueError as e:
        raise SystemExit(f"{e}; see the available types with --list")
    return archive, archive.cells[key]


def _video(path, fps):
    try:
        import imageio
    except ImportError:
        print("imageio is not installed: pip install 'darwinbot[video]'")
        print("or: pip install imageio[ffmpeg]")
        return None
    return imageio.get_writer(path, fps=fps)


def record(episode, ctrl, path, seconds, fps=30, command=0.0, speed_command=1.0):
    cpg = episode.reset(ctrl, command=command, speed=speed_command)
    dt = episode.model.opt.timestep
    every = max(1, int(round(1.0 / (fps * dt))))
    fps = 1.0 / (every * dt)       # the rate the frame step really gives
    writer = _video(path, fps)
    if writer is None:
        return
    renderer = mujoco.Renderer(episode.model, height=480, width=640)
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


def open_snapshot(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found: create it with experiments/snapshot.py")


def snapshot_robot(snap, name):
    """Model, qpos frames and fps of a named snapshot robot (default: best)."""
    if not snap["robots"]:
        raise SystemExit("the snapshot has no robots")
    if name is None:
        name = max(snap["robots"], key=lambda n: snap["robots"][n]["fitness"])
    elif name not in snap["robots"]:
        raise SystemExit(f"no {name} robot in the snapshot; see the names with --list")
    robot = snap["robots"][name]
    model = mujoco.MjModel.from_xml_string(robot["xml"])
    return name, model, robot["qpos"], robot["fps"]


def record_snapshot(model, qpos, path, fps):
    writer = _video(path, fps)
    if writer is None:
        return
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="track")
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    print(f"wrote {path} ({len(qpos)} frames, {len(qpos) / fps:.1f} s)")


def watch_snapshot(model, qpos, fps, speed=1.0):
    """Play the recorded frames in a loop, at `speed` times real time."""
    from mujoco import viewer

    data = mujoco.MjData(model)
    wall_dt = 1.0 / fps / speed
    stop = []
    previous = signal.signal(signal.SIGINT, lambda *_: stop.append(True))
    try:
        with viewer.launch_passive(model, data) as handle:
            while handle.is_running() and not stop:
                for q in qpos:
                    if not handle.is_running() or stop:
                        break
                    t0 = time.perf_counter()
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    handle.sync()
                    time.sleep(max(0.0, wall_dt - (time.perf_counter() - t0)))
            handle.close()
        # same wait as watch(): the render thread is still releasing GLFW
        time.sleep(0.5)
    finally:
        signal.signal(signal.SIGINT, previous)
    if stop:
        print("stopped")


STEER_KEYS = {263: -1.0, 262: 1.0, 265: 0.0, 264: 0.0}   # GLFW left, right, up, down
SPEED_KEYS = {266: 0.1, 267: -0.1}   # GLFW page up, page down: speed command step


def watch(episode, ctrl, speed=1.0, command=0.0, speed_command=1.0):
    """Open the viewer. The arrow keys steer while it runs: left -1, right +1,
    up or down straight. Page up and page down raise and lower the speed
    command by 0.1 (0 slowest, 1 full). The phases carry on, only the command
    changes."""
    from mujoco import viewer

    cpg = episode.reset(ctrl, command=command, speed=speed_command)
    keys = []
    dt = episode.model.opt.timestep
    wall_dt = dt / speed     # real seconds per simulation step

    # Ctrl+C must not raise inside the viewer context: the exception unwinds
    # while the render thread is still drawing, the close hangs and repeated
    # Ctrl+C ends in a segfault. A flag lets the loop exit and close cleanly.
    stop = []
    previous = signal.signal(signal.SIGINT, lambda *_: stop.append(True))
    try:
        with viewer.launch_passive(episode.model, episode.data,
                                   key_callback=keys.append) as handle:
            while handle.is_running() and not stop:
                t0 = time.perf_counter()
                while keys:
                    key = keys.pop(0)
                    if key in STEER_KEYS:
                        command = STEER_KEYS[key]
                    elif key in SPEED_KEYS:
                        speed_command = min(1.0, max(0.0, speed_command + SPEED_KEYS[key]))
                    else:
                        continue
                    cpg.params = Params(ctrl, episode.layout, command, speed_command)
                    print(f"steering {command:+.0f}, speed {speed_command:.1f}")
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


def standing(morph) -> np.ndarray:
    """A controller that holds every joint at its middle and every wheel still:
    no amplitude, offsets at the middle of their range."""
    ctrl = np.zeros(ctl.CTRL_LENGTH)
    for act in morph.layout():
        ctrl[ctl.node_base(act.slot) + ctl.OFFSET] = 0.5
    return ctrl


def show_robot(a):
    """Open a hand-written robot file in the viewer, with no archive."""
    if a.plugin:
        plugins.load(a.plugin)
    morph, _ = robot_file.load(a.robot)
    print(morpho.summary(morph))
    problems = morpho.violations(morph)
    for p in problems:
        print(f"  not buildable: {p}")
    terrain = get_terrains(a.terrain)[0]
    episode = Episode(morph, terrain)
    if not episode.buildable:
        print("  parts overlap at the start; seed.py lists them")
    print(f"terrain {terrain.name}, standing still (no controller yet)")
    if a.record:
        record(episode, standing(morph), a.record, a.seconds)
    else:
        watch(episode, standing(morph), a.speed)


def main():
    p = argparse.ArgumentParser(description="darwinbot: replay an evolved robot")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains the archive was trained with")
    p.add_argument("--robot", default=None, metavar="FILE.yaml",
                   help="show a hand-written robot file standing, without an archive")
    p.add_argument("--snapshot", default=None, metavar="FILE",
                   help="play a snapshot from experiments/snapshot.py without the archive")
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
    p.add_argument("--command", type=float, default=0.0,
                   help="steering command from -1 (left) to +1 (right); in the "
                        "viewer the arrow keys change it")
    p.add_argument("--speed-command", type=float, default=1.0,
                   help="speed command from 0 (slowest) to 1 (full, as evolved); in "
                        "the viewer page up and page down change it")
    p.add_argument("--list", action="store_true",
                   help="print every archive cell with its fitness, behaviour and body parts, then exit")
    p.add_argument("--where", default=None, metavar="EXPR",
                   help="with --list, keep the robots that match, e.g. "
                        "\"wheels=0,legs>=4,fitness>0.3\" (fields: " + ", ".join(WHERE_FIELDS) + ")")
    a = p.parse_args()
    if a.where and not a.list:
        raise SystemExit("--where needs --list")
    conds = parse_where(a.where) if a.where else []

    if a.robot:
        show_robot(a)
        return
    if a.snapshot:
        snap = open_snapshot(a.snapshot)
        robots = snap["robots"]
        if a.list:
            if conds:
                raise SystemExit("--where works on the archive list, not on --snapshot")
            print("type                       fitness  frames")
            for name, r in sorted(robots.items(), key=lambda kv: -kv[1]["fitness"]):
                print(f"{name:26s} {r['fitness']:7.2f}  {len(r['qpos']):6d}")
            return
        name, model, qpos, fps = snapshot_robot(snap, a.cell)
        print(f"{name}: {len(qpos)} frames at {fps} fps, body model {snap['body_model']}")
        if a.record:
            record_snapshot(model, qpos, a.record, fps)
        else:
            watch_snapshot(model, qpos, fps, a.speed)
        return

    if a.list:
        archive = open_archive(a.archive, a.plugin)
        names = archive.names()
        if archive.task != "forward" or archive.payload:
            print(f"task: {archive.task}, payload {archive.payload:g} kg")
        print("type                       fitness  duty  height  servos  legs    cot"
              "  wheels  feet  segments  passive  nodes")
        shown = 0
        for cell, e in sorted(archive.cells.items(), key=lambda kv: -kv[1].fitness):
            duty, height, servos, _, cot, legs = e.behaviour
            morph = morpho.decode(e.morph)
            kinds = [n.kind for n in morph.nodes.values()]
            row = dict(fitness=e.fitness, duty=duty, height=height, servos=int(servos),
                       legs=int(legs), cot=cot, wheels=morph.n_wheels,
                       feet=kinds.count("foot"), segments=morph.n_segments,
                       passive=sum(k in PASSIVE_KINDS for k in kinds), nodes=len(kinds))
            if not matches(row, conds):
                continue
            shown += 1
            print(f"{names[cell]:26s} {e.fitness:7.2f}  {duty:4.2f}  {height:6.2f}  "
                  f"{int(servos):6d}  {int(legs):4d}  {cot:5.1f}  {row['wheels']:6d}  "
                  f"{row['feet']:4d}  {row['segments']:8d}  {row['passive']:7d}  "
                  f"{row['nodes']:5d}")
        print(f"{shown} of {len(archive.cells)} robots")
        return

    archive, elite = load(a.archive, a.cell, a.plugin)
    morph = morpho.decode(elite.morph)
    print(morpho.summary(morph))
    print(f"fitness {elite.fitness:.3f}  (task {archive.task}, payload {archive.payload:g} kg)")
    for terrain, s in zip(get_terrains(archive.terrains), elite.scores):
        print(f"  {terrain.name:9s} {s:+.3f}")

    terrain = get_terrains(a.terrain)[0]
    episode = Episode(morph, terrain, payload=archive.payload)
    print(f"terrain {terrain.name}")
    if a.record:
        record(episode, elite.ctrl, a.record, a.seconds, command=a.command,
               speed_command=a.speed_command)
    else:
        watch(episode, elite.ctrl, a.speed, a.command, a.speed_command)


if __name__ == "__main__":
    main()

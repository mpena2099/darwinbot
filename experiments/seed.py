"""
experiments/seed.py
-------------------
Start an archive from a robot you wrote by hand (see docs/own-design.md).

  python experiments/seed.py --robot examples/robots/two-wheel-car.yaml --out checkpoints/car
  python experiments/train.py --resume checkpoints/car/archive.pkl --batches 30

Parts marked `lock: true` in the file are kept by evolution (single robot only).
The other way round, write an evolved robot as YAML to edit it:

  python experiments/seed.py --dump --archive checkpoints/archive.pkl --type 8servos-low-crawl
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import mujoco  # noqa: E402

from darwinbot import ablate, body_options, evolve, hardware, plugins, rules  # noqa: E402
from darwinbot import morphology as morpho  # noqa: E402
from darwinbot import robot_file  # noqa: E402
from darwinbot.fitness import task_names  # noqa: E402
from darwinbot.inner import optimise_controller  # noqa: E402
from darwinbot.sim import Episode  # noqa: E402
from darwinbot.terrain import get_terrains  # noqa: E402


def check_buildable(label, morph, terrains, seconds, payload):
    """Print every reason the body cannot be simulated and return False."""
    problems = morpho.violations(morph)
    if problems:
        print(f"{label}: not buildable")
        for p in problems:
            print(f"  - {p}")
        return False
    for t in terrains:
        ep = Episode(morph, t, seconds, payload)
        if ep.buildable:
            continue
        print(f"{label}: parts overlap at the start on terrain '{t.name}'")
        from darwinbot.builder import AssemblyRules
        data, model = mujoco.MjData(ep.model), ep.model
        flags = model.opt.disableflags
        model.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
        mujoco.mj_forward(model, data)
        hits = AssemblyRules(morph, model).clashes(model, data, fixed=True)
        model.opt.disableflags = flags
        for a, b, depth in hits[:8]:
            na = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, a) or f"geom {a}"
            nb = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b) or f"geom {b}"
            print(f"  - {na} / {nb} overlap by {depth * 1000:.1f} mm")
        return False
    return True


def dump(a):
    from replay import load
    archive, elite = load(a.archive, a.type, a.plugin)
    name = archive.names()[evolve.cell_of(elite.behaviour)]
    print(robot_file.dump(morpho.decode(elite.morph), archive.locked, name=name), end="")


def main():
    # --plugin must load before the parser checks --task and --terrains.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--plugin", default=None, metavar="FILE.py")
    early, _ = pre.parse_known_args()
    record = None
    if early.plugin:
        try:
            record = plugins.load(early.plugin)
        except OSError as e:
            raise SystemExit(str(e))

    p = argparse.ArgumentParser(description="darwinbot: seed an archive with a hand-written robot")
    p.add_argument("--robot", nargs="+", metavar="YAML", help="robot files")
    p.add_argument("--out", help="folder for archive.pkl (must not hold one yet)")
    p.add_argument("--terrains", default="flat")
    p.add_argument("--task", choices=task_names(), default="steer")
    p.add_argument("--payload", type=float, default=0.0, metavar="KG")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--variants", type=int, default=None,
                   help="randomised models per terrain (default 2)")
    p.add_argument("--profile", default=None, metavar="PATH", help="hardware profile YAML")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains to load; stored in the archive")
    p.add_argument("--body", default="", metavar="A,B", help="body limits, as in train.py")
    p.add_argument("--budget", type=int, default=400,
                   help="CMA-ES evaluations (scaled with the joint count)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dump", action="store_true",
                   help="print an archive elite as robot YAML (--archive, --type)")
    p.add_argument("--archive", default="checkpoints/archive.pkl")
    p.add_argument("--type", default=None, help="robot type from replay.py --list")
    a = p.parse_args()

    if a.profile:
        try:
            hardware.require(hardware.load(a.profile))
        except (OSError, ValueError) as e:
            raise SystemExit(str(e))
    if a.dump:
        return dump(a)
    if not a.robot or not a.out:
        raise SystemExit("give --robot FILE... and --out DIR (or --dump)")
    try:
        body = body_options.parse(a.body).to_string()
    except ValueError as e:
        raise SystemExit(str(e))
    body_options.set_active(body)

    path = os.path.join(a.out, "archive.pkl")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists, not overwriting: pick another --out")
    robots = []
    for f in a.robot:
        try:
            robots.append((f, *robot_file.load(f)))
        except (OSError, ValueError) as e:
            raise SystemExit(f"{f}: {e}")
    locks = [f for f, _, locked in robots if locked]
    if locks and len(robots) > 1:
        raise SystemExit("lock is allowed only with a single --robot "
                         f"(locks in {', '.join(locks)})")

    if a.seed is None:
        a.seed = int.from_bytes(os.urandom(4), "little") >> 1
    print(f"[darwinbot] seed={a.seed}")
    variants = 2 if a.variants is None else a.variants
    terrains = get_terrains(a.terrains, seed=a.seed)

    ok = True
    for f, morph, _ in robots:
        ok &= check_buildable(f, morph, terrains, a.seconds, a.payload)
    if not ok:
        raise SystemExit(1)

    archive = evolve.Archive(terrains=a.terrains, seconds=a.seconds, variants=variants,
                             task=a.task, payload=a.payload,
                             ablate=ablate.normalise(()), rules=rules.normalise(()),
                             body=body, profile=hardware.PROFILE,
                             locked=robots[0][2] if len(robots) == 1 else (),
                             plugin=record or {})
    t0 = time.time()
    names = []
    for k, (f, morph, _) in enumerate(robots):
        res = optimise_controller(
            morph, terrains, budget=evolve.scaled_budget(a.budget, morph),
            sigma0=0.25, seed=a.seed + k, seconds=a.seconds, variants=variants,
            task=a.task, payload=a.payload)
        genes = morpho.encode(morph)
        added = archive.insert(evolve.Elite(genes, res.ctrl, res.fitness, res.scores,
                                            0, res.behaviour))
        name = evolve.type_name(res.behaviour)
        names.append(name)
        print(f"{f}: {name}  fitness {res.fitness:.3f}"
              + ("" if added else "  (lost its cell to an earlier robot)"))
    s = archive.stats()
    s.update(added=len(archive.cells), seconds=round(time.time() - t0, 1),
             code=evolve.code_version())
    archive.history.append(s)
    archive.batch = 1
    os.makedirs(a.out, exist_ok=True)
    with open(path, "xb") as fh:
        pickle.dump(archive, fh)
    print(f"wrote {path}"
          + (f" (locked nodes {list(archive.locked)})" if archive.locked else ""))
    plug = f" --plugin {a.plugin}" if a.plugin else ""
    print(f"continue with: python experiments/train.py --resume {path} --batches 30{plug}")


if __name__ == "__main__":
    main()

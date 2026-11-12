"""
experiments/train.py
--------------------
Run MAP-Elites over body plans, with CMA-ES learning each body's gait.

  python experiments/train.py                          # flat, defaults
  python experiments/train.py --device auto            # benchmark, use the fastest
  python experiments/train.py --terrains terrain       # flat + rough + steps
  python experiments/train.py --batches 60 --workers 8
  python experiments/train.py --task turn --terrains step --payload 0.3
  python experiments/train.py --resume checkpoints/archive.pkl --batches 20
  python experiments/train.py --preset turn,step,payload --batches 5

A preset is a ready task (see PRESETS); several run one after the other, each
in its own folder (checkpoints/<preset>), and a preset whose folder already
holds an archive is resumed, so the same command run again trains further.
"""

import argparse
import os
import sys

from darwinbot import ablate, body_options, evolve, hardware, plugins, rules
from darwinbot.evolve import load_archive
from darwinbot.fitness import task_names
from darwinbot.morphology import decode, summary

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark  # noqa: E402  (experiments/benchmark.py, the device measurement)

PRESETS = {
    "steer":   dict(task="steer", terrains="terrain", payload=0.0),
    "forward": dict(task="forward", terrains="terrain", payload=0.0),
    "turn":    dict(task="turn", terrains="flat", payload=0.0),
    "step":    dict(task="steer", terrains="step", payload=0.0),
    "payload": dict(task="steer", terrains="terrain", payload=0.3),
}


def resolve_device(device, workers, cache_path=None):
    """--device auto: the device measured on this machine (experiments/benchmark.py),
    read from the cache, or measured and saved first when the cache is missing.
    A given --workers wins over the cached one; any other device is returned as
    it is, without touching the cache."""
    if device != "auto":
        return device, workers
    result = benchmark.load(cache_path)
    if result is None:
        result = benchmark.measure()
        benchmark.save(result, cache_path)
    picked = result.get("device", "cpu")
    if workers is None:
        workers = result.get("workers")
    print(f"[darwinbot] --device auto: {picked}"
          f"{'' if workers is None else f' with {workers} workers'}, "
          f"benchmarked {result.get('date', '?')} -> {benchmark.cache_path(cache_path)}")
    return picked, workers


def check_kit_profile(path):
    """Sanity-check a --profile kit before the run; warnings are printed and
    errors stop it (tools/check_kit.py)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from check_kit import check
    errors, warnings, _ = check(path)
    for message in warnings:
        print(f"[kit] warning: {message}")
    if errors:
        raise SystemExit("hardware profile check failed:\n  " +
                         "\n  ".join(errors))


def main():
    # --plugin must load before the parser checks --task and --terrains, which
    # the plugin may add to.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--plugin", default=None, metavar="FILE.py")
    early, _ = pre.parse_known_args()
    record = None
    if early.plugin:
        try:
            record = plugins.load(early.plugin)
        except OSError as e:
            raise SystemExit(str(e))

    p = argparse.ArgumentParser(description="darwinbot: evolve bodies and gaits")
    p.add_argument("--terrains", default="flat",
                   help="preset (flat, terrain, all) or comma list of terrain names")
    p.add_argument("--init", type=int, default=24, help="random bodies to start with")
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--budget-init", type=int, default=400,
                   help="CMA-ES evaluations for a body with a random controller")
    p.add_argument("--budget-child", type=int, default=120,
                   help="CMA-ES evaluations for a body with an inherited controller")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (default: a new one every run, printed so "
                        "the run can be repeated)")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--device", choices=("cpu", "gpu", "warp", "auto"), default="cpu",
                   help="simulate on the CPU (default), with MJX (gpu), with "
                        "MuJoCo Warp (warp), or auto: benchmark this machine "
                        "once and pick the fastest; with warp the GPU only "
                        "searches, the archive is still scored on the CPU")
    p.add_argument("--popsize", type=int, default=None,
                   help="CMA-ES population (gpu default: 64)")
    p.add_argument("--seconds", type=float, default=8.0,
                   help="simulated seconds per episode, 8 by default (ignored on --resume)")
    p.add_argument("--variants", type=int, default=None,
                   help="randomised models per terrain for every evaluation "
                        "(default: 2 on cpu and warp, 0 on gpu; ignored on --resume)")
    p.add_argument("--parent-selection", choices=evolve.PARENT_SELECTIONS,
                   default="novelty",
                   help="parents drawn by novelty (default) or uniformly")
    p.add_argument("--task", choices=task_names(), default="steer",
                   help="what the score rewards: walking straight and turning "
                        "when told (steer, default), walking forward only, or "
                        "turning in place (ignored on --resume)")
    p.add_argument("--payload", type=float, default=0.0, metavar="KG",
                   help="extra mass carried on the torso (ignored on --resume)")
    p.add_argument("--resume", default=None)
    p.add_argument("--profile", default=None, metavar="PATH",
                   help="hardware profile YAML (default: the built-in STS3215 12 V one; "
                        "stored in the archive; see docs/hardware-profiles.md)")
    p.add_argument("--plugin", default=None, metavar="FILE.py",
                   help="external tasks and terrains to load; stored in the archive "
                        "(see docs/plugins.md)")
    p.add_argument("--preset", default=None,
                   help="comma list of ready tasks run in sequence: "
                        + ", ".join(PRESETS) + " (each in checkpoints/<preset>)")
    p.add_argument("--ablate", action="append", default=[], metavar="NAME",
                   choices=ablate.NAMES,
                   help="experiment: switch one realism rule off, repeatable ("
                        + ", ".join(ablate.NAMES) + "); stored in the archive, CPU only")
    p.add_argument("--rules", default="", metavar="A,B",
                   help="experiment: optional score rules to switch on, comma list ("
                        + ", ".join(rules.NAMES) + "); stored in the archive")
    p.add_argument("--body", default="", metavar="A,B",
                   help="limits on the bodies evolution may build, comma list of "
                        "no-wheels, no-passive, no-segments, max-servos=N, "
                        "legs=A..B, max-size=MM, bed=MM; stored in the archive")
    a = p.parse_args()
    a.rules = tuple(n for n in a.rules.split(",") if n)
    try:
        rules.normalise(a.rules)
        body_options.parse(a.body)
    except ValueError as e:
        raise SystemExit(str(e))

    a.device, a.workers = resolve_device(a.device, a.workers)
    wanted = None
    if a.profile:
        check_kit_profile(a.profile)
        try:
            wanted = hardware.load(a.profile)
        except (OSError, ValueError) as e:
            raise SystemExit(str(e))
        hardware.require(wanted)        # re-runs this command under that profile

    def check_profile(path):
        """A run that resumes an archive trains on the archive's hardware and plugin."""
        stored = load_archive(path)
        if wanted is not None and hardware.normalize(stored.profile) != wanted:
            raise SystemExit(f"{path} was trained on another hardware profile than "
                             f"--profile {a.profile}; drop --profile to keep the archive's")
        plugins.require(stored, a.plugin)
        hardware.require(stored.profile)

    if a.resume:
        check_profile(a.resume)

    if a.device == "gpu":
        try:
            import jax  # noqa: F401
            from mujoco import mjx  # noqa: F401
        except ImportError:
            raise SystemExit("--device gpu needs JAX and MJX: "
                             "pip install -e '.[gpu]'")
    if a.device == "warp":
        try:
            import mujoco_warp  # noqa: F401
        except ImportError:
            raise SystemExit("--device warp needs MuJoCo Warp: "
                             "pip install -e '.[warp]'")

    if a.seed is None:
        a.seed = int.from_bytes(os.urandom(4), "little") >> 1
    print(f"[darwinbot] seed={a.seed}")
    variants = a.variants if a.variants is not None else (0 if a.device == "gpu" else 2)
    common = dict(n_init=a.init, n_batches=a.batches, batch_size=a.batch_size,
                  budget_init=a.budget_init, budget_child=a.budget_child,
                  workers=a.workers, seed=a.seed, episode_seconds=a.seconds,
                  device=a.device, popsize=a.popsize, variants=variants,
                  parent_selection=a.parent_selection, ablations=tuple(a.ablate), score_rules=a.rules,
                  body=a.body, plugin=record)
    forward = "forward" in a.ablate     # the forward task replaces steer

    if not a.preset:
        report(evolve.run(terrains=a.terrains, out_dir=a.out, resume=a.resume,
                          task="forward" if forward else a.task,
                          payload=a.payload, **common))
        return

    names = [n.strip() for n in a.preset.split(",") if n.strip()]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown preset {unknown}, expected some of {list(PRESETS)}")
    for name in names:
        out = os.path.join(a.out, name)
        path = os.path.join(out, "archive.pkl")
        resume = path if os.path.exists(path) else None
        if resume:
            check_profile(resume)
        print(f"\n=== preset {name}: {PRESETS[name]} -> {out}"
              f"{' (resuming)' if resume else ''}")
        spec = dict(PRESETS[name])
        if forward:
            spec["task"] = "forward"
        archive = evolve.run(out_dir=out, resume=resume, **spec, **common)
        report(archive)
        if getattr(archive, "interrupted", False):
            print("Ctrl+C: the remaining presets were not run")
            return


def report(archive):
    if not archive.cells:
        return
    best = archive.best()
    print(f"\nBest fitness {best.fitness:.3f}  scores {[round(s, 2) for s in best.scores]}")
    print(summary(decode(best.morph)))


if __name__ == "__main__":
    main()

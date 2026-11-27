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
    p.add_argument("--inner-strategy", choices=("default", "diagonal", "focused",
                                                "diagonal+focused"),
                   default="default",
                   help="inner-loop CMA-ES (default: plain CMA-ES; diagonal: "
                        "diagonal covariance; focused: wider initial steps where the "
                        "body changed; diagonal+focused: both); not stored in the archive")
    p.add_argument("--variant-mode", choices=evolve.VARIANT_MODES, default="all",
                   help="all (default): every candidate on the whole variant set; "
                        "first: one variant in the loop, one full re-score at the end")
    p.add_argument("--protect-new", type=float, default=1.0, metavar="FACTOR",
                   help="multiply the inner-loop budget of a child whose body changed "
                        "a lot (a graft, two genome slots or an added/removed node), "
                        "so a new body is not judged on a poorly tuned controller; "
                        "1.0 (default) turns it off; not stored in the archive")
    p.add_argument("--deterministic", action="store_true",
                   help="take each batch's results in submission order instead of "
                        "completion order, so the same --seed and --workers repeat a "
                        "run exactly; a little slower; not stored in the archive. "
                        "--device gpu always runs in submission order, so it is "
                        "deterministic with or without this flag")
    p.add_argument("--mirror-p", type=float, default=0.0, metavar="P",
                   help="share of mutations that copy the limb that changed onto its "
                        "mirror slot, so symmetric bodies can arise and compete; "
                        "0 (default) turns it off and changes nothing; not stored in "
                        "the archive and never part of the score")
    p.add_argument("--mutation-bias", choices=("default", "remove"), default="default",
                   help="weigh the add/remove mutation operators towards smaller "
                        "bodies (add 0.10, remove 0.25 instead of 0.20/0.10); "
                        "default leaves all weights as they are; stored in the archive")
    p.add_argument("--tie-parts", type=float, default=0.0, metavar="EPS",
                   help="let a child with fewer body parts take an occupied cell when "
                        "its objective is within EPS relative of the incumbent's; "
                        "0 (default) turns it off; stored in the archive")
    p.add_argument("--screen", type=float, default=1.0, metavar="SHARE",
                   help="cheap evaluation first: every child whose body changed gets "
                        "one quick score (its inherited controller, one flat episode), "
                        "and only the best SHARE of them (0 to 1) go on to the "
                        "inner loop; 1.0 (default) turns it off; CPU only")
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
    p.add_argument("--rules", default=None, metavar="A,B",
                   help="experiment: optional score rules to switch on, comma list ("
                        + ", ".join(rules.NAMES) + "); stored in the archive, and a "
                        "resume must repeat the archive's (or give none)")
    p.add_argument("--basal-scale", type=float, default=None, metavar="SCALE",
                   help="multiply the basal rule's per-servo and per-kg costs "
                        "(default 1); stored in the archive")
    p.add_argument("--basal-mode", choices=rules.BASAL_MODES, default=None,
                   help="absolute (default): subtract the per-servo and per-kg "
                        "costs; relative: keep (1 - f) of a positive score, "
                        "f = servo share * servos + part share * passive parts, "
                        "capped at 0.8; stored in the archive")
    p.add_argument("--basal-servo-share", type=float, default=None, metavar="SHARE",
                   help="score fraction charged per servo by --basal-mode "
                        "relative (default 0.04); stored in the archive")
    p.add_argument("--basal-part-share", type=float, default=None, metavar="SHARE",
                   help="score fraction charged per passive part by "
                        "--basal-mode relative (default 0.02); stored in the archive")
    p.add_argument("--turn-scale", type=float, default=None, metavar="M/RAD",
                   help="metres per radian of asked-for turn (default 0.2); "
                        "stored in the archive")
    p.add_argument("--turn-in-place", action="store_true", default=None,
                   help="let a pivot earn turn credit, capped by the straight "
                        "walk (default: off); stored in the archive")
    p.add_argument("--turn-tight", type=float, default=None, metavar="W",
                   help="tight-turn bonus: a turn's credit grows by up to W times as its "
                        "radius falls below 0.5 m (default 0, off); stored in the archive")
    p.add_argument("--turn-cap-deg", type=float, default=None, metavar="DEG",
                   help="most turn credited in one episode (default 90 degrees); "
                        "stored in the archive")
    p.add_argument("--body", default="", metavar="A,B",
                   help="limits on the bodies evolution may build, comma list of "
                        "no-wheels, no-passive, no-segments, max-servos=N, "
                        "legs=A..B, max-size=MM, bed=MM; stored in the archive")
    p.add_argument("--axes", choices=evolve.AXES, default="default",
                   help="archive descriptors: default (five), body (five plus the "
                        "share of ticks the torso touches the ground) or r1 (the four "
                        "release-1 descriptors, no servo count, 500 cells); stored in "
                        "the archive, new archives only")
    a = p.parse_args()
    rules_given = a.rules is not None          # None: keep the archive's rules on resume
    a.rules = tuple(n for n in (a.rules or "").split(",") if n)
    try:
        rules.normalise(a.rules)
        body_options.parse(a.body)
    except ValueError as e:
        raise SystemExit(str(e))
    if a.protect_new < 1.0:
        raise SystemExit("--protect-new must be >= 1.0 (1.0 turns it off)")
    if not 0.0 <= a.mirror_p <= 1.0:
        raise SystemExit("--mirror-p must be in [0, 1] (0 turns it off)")
    if a.tie_parts < 0.0:
        raise SystemExit("--tie-parts must be >= 0 (0 turns it off)")
    if not 0.0 < a.screen <= 1.0:
        raise SystemExit("--screen must be in (0, 1] (1.0 turns it off)")

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
    score = evolve.score_options(a.basal_scale, a.turn_scale, a.turn_in_place,
                                 a.turn_cap_deg, a.basal_mode, a.basal_servo_share,
                                 a.basal_part_share, turn_tight=a.turn_tight)
    common = dict(n_init=a.init, n_batches=a.batches, batch_size=a.batch_size,
                  budget_init=a.budget_init, budget_child=a.budget_child,
                  workers=a.workers, seed=a.seed, episode_seconds=a.seconds,
                  device=a.device, popsize=a.popsize, variants=variants,
                  parent_selection=a.parent_selection, ablations=tuple(a.ablate),
                  score_rules=a.rules if rules_given else None, score=score,
                  body=a.body, axes=a.axes, plugin=record,
                  inner_strategy=None if a.inner_strategy == "default" else a.inner_strategy,
                  variant_mode=a.variant_mode, protect_new=a.protect_new,
                  mirror_p=a.mirror_p, mutation_bias=a.mutation_bias,
                  tie_parts=a.tie_parts, deterministic=a.deterministic,
                  screen=a.screen)
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

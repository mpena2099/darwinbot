"""
experiments/visualise.py
------------------------
Loads the best individual from a checkpoint and displays it
in the MuJoCo interactive viewer.

Usage
-----
python experiments/visualise.py --checkpoint checkpoints/gen_0050.pkl
python experiments/visualise.py --checkpoint checkpoints/gen_0050.pkl --scenario obstacles
python experiments/visualise.py --checkpoint checkpoints/gen_0050.pkl --speed 0.3
"""

import argparse
import pickle
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from darwinbot.genome import decode, genome_summary
from darwinbot.mjcf_builder import write_mjcf
from darwinbot.scenarios.registry import REGISTRY


def load_best(checkpoint_path: str):
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    hof = ckpt.get("hof")
    if hof and len(hof) > 0:
        return np.array(hof[0]), hof[0].fitness.values[0], ckpt["generation"]

    pop  = ckpt["population"]
    best = max(pop, key=lambda ind: ind.fitness.values[0]
               if ind.fitness.valid else float("-inf"))
    return np.array(best), best.fitness.values[0], ckpt["generation"]


def main():
    parser = argparse.ArgumentParser(
        description="darwinbot — visualise best robot from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario",   type=str, default="flat",
                        help=f"Available: {[n for n,_,_,_ in REGISTRY]}")
    parser.add_argument("--speed",      type=float, default=1.0,
                        help="Playback speed multiplier (e.g. 0.3 for slow-mo)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[visualise] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    genes, fitness, generation = load_best(args.checkpoint)

    print("=" * 60)
    print("  darwinbot — visualise")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Generation : {generation}")
    print(f"  Fitness    : {fitness:.4f}")
    print(f"  Scenario   : {args.scenario}")
    print(f"  Speed      : {args.speed}x")
    print()
    print(genome_summary(genes))
    print()

    # Find scenario
    entry = next(((gx, bf) for n, gx, bf, _ in REGISTRY
                  if n == args.scenario), None)
    if entry is None:
        print(f"[visualise] Unknown scenario '{args.scenario}'. "
              f"Available: {[n for n,_,_,_ in REGISTRY]}")
        sys.exit(1)

    get_xml_fn, build_fn = entry

    import mujoco
    import mujoco.viewer
    import time

    robot = decode(genes)
    mjcf  = write_mjcf(robot, scenario_xml=get_xml_fn())

    try:
        model = mujoco.MjModel.from_xml_path(mjcf)
        data  = mujoco.MjData(model)
        info  = build_fn(model, data)

        data.qpos[:3] = info.spawn_pos
        mujoco.mj_forward(model, data)

        from darwinbot.simulator import SIM_HZ, _apply_control

        print("[visualise] MuJoCo viewer open.")
        print("            Use mouse to rotate/zoom. Close window to exit.")

        step  = 0
        delay = (1.0 / SIM_HZ) / args.speed

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t = step / SIM_HZ
                _apply_control(model, data, robot, t)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(delay)
                step += 1

    except KeyboardInterrupt:
        print("\n[visualise] Interrupted.")
    finally:
        try:
            os.remove(mjcf)
        except OSError:
            pass


if __name__ == "__main__":
    main()

"""
experiments/visualise.py
------------------------
Loads the best individual from a checkpoint and displays it
in the PyBullet GUI.

Usage
-----
# Visualise best robot from a specific checkpoint
python experiments/visualise.py --checkpoint checkpoints/gen_0050.pkl

# Choose which scenario to display in (default: flat)
python experiments/visualise.py --checkpoint checkpoints/gen_0050.pkl --scenario obstacles

# Slow down playback (default speed is real-time)
python experiments/visualise.py --checkpoint checkpoints/gen_0050.pkl --speed 0.3
"""

import argparse
import pickle
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from darwinbot.genome import decode, genome_summary
from darwinbot.urdf_builder import write_urdf
from darwinbot.simulator import (
    _connect, _load_robot, _get_revolute_joints,
    _apply_control, SIM_HZ,
)
from darwinbot.scenarios.registry import REGISTRY


def load_best(checkpoint_path: str):
    """Load the best individual from a checkpoint file."""
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    hof = ckpt.get("hof")
    if hof and len(hof) > 0:
        return np.array(hof[0]), hof[0].fitness.values[0], ckpt["generation"]

    # Fallback: best in population
    pop = ckpt["population"]
    best = max(pop, key=lambda ind: ind.fitness.values[0]
               if ind.fitness.valid else float("-inf"))
    return np.array(best), best.fitness.values[0], ckpt["generation"]


def main():
    parser = argparse.ArgumentParser(
        description="darwinbot — visualise best robot from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pkl file")
    parser.add_argument("--scenario", type=str, default="flat",
                        help=(
                            f"Scenario to display in. "
                            f"Available: {[n for n, _, _ in REGISTRY]}"
                        ))
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0, try 0.3 for slow-mo)")
    args = parser.parse_args()

    # ── Load checkpoint ───────────────────────────────────────────────────────
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

    # ── Find scenario builder ─────────────────────────────────────────────────
    builder = None
    for name, build_fn, _ in REGISTRY:
        if name == args.scenario:
            builder = build_fn
            break

    if builder is None:
        print(f"[visualise] Unknown scenario '{args.scenario}'. "
              f"Available: {[n for n, _, _ in REGISTRY]}")
        sys.exit(1)

    # ── Launch PyBullet GUI ───────────────────────────────────────────────────
    import pybullet as p

    robot  = decode(genes)
    urdf   = write_urdf(robot)
    client = _connect(visualise=True)

    info      = builder(client)
    robot_id  = _load_robot(urdf, info.spawn_pos, info.spawn_orn, client)
    joint_map = _get_revolute_joints(robot_id, client)

    print("[visualise] PyBullet GUI open — close the window to exit.")
    print("            Press Ctrl+C in terminal to force quit.")

    step     = 0
    delay    = (1.0 / SIM_HZ) / args.speed

    try:
        while p.isConnected(client):
            t = step / SIM_HZ
            _apply_control(robot_id, joint_map, robot, t, client)
            p.stepSimulation(physicsClientId=client)
            time.sleep(delay)
            step += 1
    except KeyboardInterrupt:
        print("\n[visualise] Interrupted.")
    finally:
        try:
            p.disconnect(client)
        except Exception:
            pass
        try:
            os.remove(urdf)
        except OSError:
            pass


if __name__ == "__main__":
    main()

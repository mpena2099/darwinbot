"""
experiments/run_basic.py
------------------------
Entry point for a darwinbot evolution run.

Usage examples
--------------
# Phase 1 only (flat floor) — default
python experiments/run_basic.py

# Phase 2 (flat + surfaces)
python experiments/run_basic.py --scenarios phase2

# Phase 3 (flat + surfaces + obstacles)
python experiments/run_basic.py --scenarios phase3

# All scenarios
python experiments/run_basic.py --scenarios all

# Custom subset
python experiments/run_basic.py --scenarios flat,obstacles

# Resume from checkpoint
python experiments/run_basic.py --resume checkpoints/gen_0050.pkl --scenarios phase2

# Visualise best individual in PyBullet GUI
python experiments/run_basic.py --visualise --scenarios flat
"""

import argparse
import functools
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from darwinbot.scenarios.registry import get_active_scenarios, list_scenarios
from darwinbot.fitness import evaluate
from darwinbot.evolution import run_evolution, toolbox
from darwinbot.genome import decode, genome_summary
from darwinbot.simulator import simulate


def main():
    parser = argparse.ArgumentParser(
        description="darwinbot — co-evolution of morphology + locomotion")
    parser.add_argument("--pop",       type=int,   default=40)
    parser.add_argument("--gen",       type=int,   default=50)
    parser.add_argument("--workers",   type=int,   default=None)
    parser.add_argument("--resume",    type=str,   default=None)
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument(
        "--scenarios", type=str, default="phase1",
        help=(
            "Scenario preset or comma-separated names. "
            f"Presets: phase1 phase2 phase3 phase4 all. "
            f"Scenarios: {', '.join(list_scenarios())}"
        ),
    )
    args = parser.parse_args()

    active = get_active_scenarios(args.scenarios)
    print("=" * 60)
    print("  darwinbot — co-evolution of morphology + locomotion")
    print("=" * 60)
    print(f"  Scenarios : {[n for n, _, _ in active]}")
    print(f"  Weights   : {[round(w, 3) for _, _, w in active]}")
    print()

    # Bind active scenarios into the evaluate function and register with DEAP
    eval_fn = functools.partial(evaluate, active_scenarios=active)
    toolbox.register("evaluate", eval_fn)

    population, logbook, hof = run_evolution(
        pop_size      = args.pop,
        n_generations = args.gen,
        n_workers     = args.workers,
        resume_from   = args.resume,
    )

    best_genes = np.array(hof[0])
    print("\n" + "=" * 60)
    print("  Best individual found:")
    print("=" * 60)
    print(genome_summary(best_genes))
    print(f"\n  Fitness: {hof[0].fitness.values[0]:.4f}")

    if args.visualise:
        print("\n[darwinbot] Launching PyBullet GUI — close the window to exit.")
        import time
        import pybullet as p
        from darwinbot.scenarios.flat import build as flat_build
        from darwinbot.urdf_builder import write_urdf
        from darwinbot.simulator import (
            _connect, _load_robot, _get_revolute_joints,
            _apply_control, SIM_HZ,
        )

        robot  = decode(best_genes)
        urdf   = write_urdf(robot)
        client = _connect(visualise=True)
        flat_build(client)
        robot_id  = _load_robot(urdf, (0, 0, 0.35), (0, 0, 0), client)
        joint_map = _get_revolute_joints(robot_id, client)

        step = 0
        while p.isConnected(client):
            t = step / SIM_HZ
            _apply_control(robot_id, joint_map, robot, t, client)
            p.stepSimulation(physicsClientId=client)
            time.sleep(1.0 / SIM_HZ)
            step += 1

        try:
            os.remove(urdf)
        except OSError:
            pass


if __name__ == "__main__":
    main()

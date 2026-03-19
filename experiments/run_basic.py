"""
experiments/run_basic.py
------------------------
Entry point for a basic darwinbot evolution run.

Usage:
    python experiments/run_basic.py
    python experiments/run_basic.py --resume checkpoints/gen_0050.pkl
    python experiments/run_basic.py --visualise   (shows best individual in GUI)
"""

import argparse
import sys
import os
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from darwinbot.evolution import run_evolution
from darwinbot.genome import decode, genome_summary
from darwinbot.simulator import simulate


def main():
    parser = argparse.ArgumentParser(description="darwinbot — basic evolution run")
    parser.add_argument("--pop",        type=int,   default=40,
                        help="Population size (default: 40)")
    parser.add_argument("--gen",        type=int,   default=50,
                        help="Number of generations (default: 50)")
    parser.add_argument("--workers",    type=int,   default=None,
                        help="Parallel workers (default: cpu_count - 1)")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint .pkl to resume from")
    parser.add_argument("--visualise",  action="store_true",
                        help="Visualise best individual in PyBullet GUI after run")
    args = parser.parse_args()

    print("=" * 60)
    print("  darwinbot — co-evolution of morphology + locomotion")
    print("=" * 60)

    population, logbook, hof = run_evolution(
        pop_size       = args.pop,
        n_generations  = args.gen,
        n_workers      = args.workers,
        resume_from    = args.resume,
    )

    # -- Report best individual --
    best_genes = np.array(hof[0])
    print("\n" + "=" * 60)
    print("  Best individual found:")
    print("=" * 60)
    print(genome_summary(best_genes))
    print(f"\n  Fitness: {hof[0].fitness.values[0]:.4f}")

    # -- Visualise in GUI --
    if args.visualise:
        print("\n[darwinbot] Launching PyBullet GUI — close the window to exit.")
        import time
        import pybullet as p
        from darwinbot.genome import decode
        from darwinbot.urdf_builder import write_urdf
        from darwinbot.simulator import _connect, _load_robot, _get_joint_map, _apply_control, SIM_HZ

        robot  = decode(best_genes)
        urdf   = write_urdf(robot)
        client = _connect(visualise=True)
        robot_id  = _load_robot(urdf, client)
        joint_map = _get_joint_map(robot_id, client)

        step = 0
        while p.isConnected(client):
            t = step / SIM_HZ
            _apply_control(robot_id, joint_map, robot, t, client)
            p.stepSimulation(physicsClientId=client)
            time.sleep(1.0 / SIM_HZ)
            step += 1

        os.remove(urdf)


if __name__ == "__main__":
    main()

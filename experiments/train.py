"""
experiments/train.py
--------------------
Runs the evolutionary loop and saves checkpoints.

Usage
-----
# Phase 1 — flat floor (default)
python experiments/train.py

# Phase 2 — flat + varied surfaces
python experiments/train.py --scenarios phase2

# Phase 3 — flat + surfaces + obstacles
python experiments/train.py --scenarios phase3

# All scenarios
python experiments/train.py --scenarios all

# Custom population and generation count
python experiments/train.py --pop 80 --gen 200 --workers 6

# Resume from a previous checkpoint
python experiments/train.py --resume checkpoints/gen_0050.pkl --scenarios phase2
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
from darwinbot.genome import genome_summary


def main():
    parser = argparse.ArgumentParser(
        description="darwinbot — train: evolve robot morphology + locomotion")
    parser.add_argument("--pop",      type=int, default=40,
                        help="Population size (default: 40)")
    parser.add_argument("--gen",      type=int, default=50,
                        help="Number of generations (default: 50)")
    parser.add_argument("--workers",  type=int, default=None,
                        help="Parallel workers (default: cpu_count - 1)")
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint .pkl to resume from")
    parser.add_argument("--scenarios", type=str, default="phase1",
                        help=(
                            "Scenario preset or comma-separated names. "
                            f"Presets: phase1 phase2 phase3 phase4 all. "
                            f"Available: {', '.join(list_scenarios())}"
                        ))
    args = parser.parse_args()

    active = get_active_scenarios(args.scenarios)

    print("=" * 60)
    print("  darwinbot — train")
    print("=" * 60)
    print(f"  Scenarios  : {[n for n, _, _, _ in active]}")
    print(f"  Weights    : {[round(w, 3) for _, _, _, w in active]}")
    print(f"  Pop size   : {args.pop}")
    print(f"  Generations: {args.gen}")
    print()

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
    print(f"\n  Fitness    : {hof[0].fitness.values[0]:.4f}")
    print(f"  Checkpoints: checkpoints/")


if __name__ == "__main__":
    main()

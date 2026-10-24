"""
experiments/train.py
--------------------
Run MAP-Elites over body plans, with CMA-ES learning each body's gait.

  python experiments/train.py                          # flat, defaults
  python experiments/train.py --terrains terrain       # flat + rough + steps
  python experiments/train.py --batches 60 --workers 8
  python experiments/train.py --resume checkpoints/archive.pkl --batches 20
"""

import argparse

from darwinbot import evolve
from darwinbot.morphology import decode, summary


def main():
    p = argparse.ArgumentParser(description="darwinbot: evolve bodies and gaits")
    p.add_argument("--terrains", default="flat",
                   help="preset (flat, terrain, all) or comma list of terrain names")
    p.add_argument("--init", type=int, default=24, help="random bodies to start with")
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--budget-init", type=int, default=400,
                   help="CMA-ES evaluations for a body with a random controller")
    p.add_argument("--budget-child", type=int, default=200,
                   help="CMA-ES evaluations for a body with an inherited controller")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--resume", default=None)
    a = p.parse_args()

    archive = evolve.run(
        terrains=a.terrains, n_init=a.init, n_batches=a.batches,
        batch_size=a.batch_size, budget_init=a.budget_init,
        budget_child=a.budget_child, workers=a.workers, seed=a.seed,
        out_dir=a.out, resume=a.resume,
    )
    best = archive.best()
    print(f"\nBest fitness {best.fitness:.3f}  scores {[round(s, 2) for s in best.scores]}")
    print(summary(decode(best.morph)))


if __name__ == "__main__":
    main()

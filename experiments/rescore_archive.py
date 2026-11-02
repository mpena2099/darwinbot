"""
Score every elite of an archive again under the current fitness rules, keeping
bodies, controllers, cells and lineage. Used after a fitness fix, so a run can
continue from its archive instead of starting over.

    python experiments/rescore_archive.py checkpoints/steer/archive.pkl --out rescored.pkl

Each elite runs on the archive's terrains (the fixed layouts) and, when the run
used randomised physics, on that many variants drawn from --seed. Cells do not
change: the behaviour descriptors come from the nominal flat episode, which the
fitness rules do not touch.
"""

import argparse
import os
import pickle
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np   # noqa: E402

from darwinbot import morphology as morpho, randomize   # noqa: E402
from darwinbot.evolve import Archive, Elite   # noqa: E402
from darwinbot.fitness import UNSTABLE_SCORE   # noqa: E402
from darwinbot.inner import evaluate   # noqa: E402
from darwinbot.sim import Episode   # noqa: E402
from darwinbot.terrain import get_terrains   # noqa: E402


def _score(job):
    morph_genes, ctrl, terrains, seconds, variants, task, payload, seed = job
    morph = morpho.decode(morph_genes)
    episodes = [Episode(morph, t, seconds, payload) for t in get_terrains(terrains)]
    if not episodes[0].buildable:
        return UNSTABLE_SCORE, [UNSTABLE_SCORE] * len(episodes)
    var_sets = None
    if variants > 0:
        rng = np.random.default_rng(seed)
        var_sets = [[randomize.perturb(ep.model, rng) for _ in range(variants)]
                    for ep in episodes]
    fit, scores, _ = evaluate(episodes, ctrl, var_sets, task)
    return fit, scores


def rescore(archive: Archive, workers: int, seed: int = 0,
            note: str = "scoring rules changed") -> Archive:
    cells = list(archive.cells.values())
    jobs = [(e.morph, e.ctrl, archive.terrains, archive.seconds, archive.variants,
             archive.task, archive.payload, seed) for e in cells]
    with Pool(workers) as pool:
        results = pool.map(_score, jobs)
    out = pickle.loads(pickle.dumps(archive))
    out._grid = None
    for e, (fit, scores) in zip(cells, results):
        out.insert(Elite(e.morph, e.ctrl, fit, scores, e.born, e.behaviour))
    out.lineage = dict(archive.lineage)
    s = out.stats()
    s.update(added=0, seconds=0.0, rescored=note)
    out.history = list(archive.history) + [s]
    return out


def main():
    p = argparse.ArgumentParser(description="darwinbot: rescore an archive")
    p.add_argument("archive")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0, help="seed of the randomised variants")
    p.add_argument("--note", default="scoring rules changed",
                   help="label of the rescoring in the progress plot")
    a = p.parse_args()
    with open(a.archive, "rb") as f:
        archive = pickle.load(f)
    before = archive.stats()
    out = rescore(archive, a.workers, a.seed, a.note)
    after = out.stats()
    with open(a.out + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(a.out + ".tmp", a.out)
    print(f"cells {before['cells']} -> {after['cells']}, best {before['best']:.2f} -> "
          f"{after['best']:.2f}, QD-score {before['qd_score']:.1f} -> {after['qd_score']:.1f}")


if __name__ == "__main__":
    main()

"""
progressive_stairs.py
---------------------
A plugin terrain: a staircase whose risers grow from 4 cm to a house step
(17 cm), on 28 cm treads, so a run shows how far up each body gets.

  python experiments/train.py --plugin examples/plugins/progressive_stairs.py \
      --terrains flat,progressive-stairs --task forward --seconds 10
"""

from darwinbot import terrain as te

RISERS = (0.04, 0.08, 0.12, 0.17)   # m, height of each step over the one before
TREAD = 0.28                        # m, depth of each step
FIRST = 0.6                         # m, x of the first riser (spawn at x = 0)


def progressive_stairs() -> te.Terrain:
    geoms = [te._plane()]
    top = FIRST + len(RISERS) * TREAD + 2.0    # every step runs on past the top
    h = 0.0
    for i, rise in enumerate(RISERS):
        x0 = FIRST + i * TREAD
        h += rise
        geoms.append(te._box((x0 + top) / 2, 0, h / 2, (top - x0) / 2, 2, h / 2))
    return te.Terrain("progressive-stairs", world="".join(geoms))


te.register_terrain("progressive-stairs", progressive_stairs)

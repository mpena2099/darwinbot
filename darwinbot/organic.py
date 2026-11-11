"""
organic.py
----------
Optional export step: hollow the flat printed links
with 2D topology optimisation, after evolution and outside training.

Evolution decides each part's size and its interfaces; this only removes
material from the middle. The plate is sampled on a grid at mid-thickness,
optimised by topology.simp with the servo-horn end clamped and an in-plane
load at the far end (across the part and along it), and the pockets where the
optimiser left no material are cut through the original part. A rim along
every edge and a solid boss around both ends always stay, so holes, outline
and thickness are those of the original part.

The load case is in-plane only. Out-of-plane bending is carried by the kept
rim, not optimised, and a hollowed robot is lighter than the one evolution
scored: check a build before trusting it.
"""

import math

import numpy as np
from build123d import Face, Part, Plane, Vector, Wire, extrude, section
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt

from darwinbot import topology

CELL_MM = 1.0        # grid step of the sampled plate
RIM_MM = 2.5         # solid band kept along every edge and hole
VOLFRAC = 0.55       # share of the plate's area left after hollowing
THRESHOLD = 0.5      # density above which a cell counts as material
MIN_POCKET_MM2 = 6.0  # smaller pockets are left filled


def _outline(wire, step: float) -> np.ndarray:
    """Points along a closed wire, about `step` apart."""
    pts = []
    for e in wire.edges():
        n = max(2, int(math.ceil(e.length / step)))
        pts += [(v.X, v.Y) for v in e.positions(np.linspace(0.0, 1.0, n, endpoint=False))]
    return np.array(pts)


def _grid(part: Part, cell: float):
    """Solid mask of the part's mid-thickness section, and the grid origin."""
    box = part.bounding_box()
    z = (box.min.Z + box.max.Z) / 2
    nx = int(math.ceil((box.max.X - box.min.X) / cell))
    ny = int(math.ceil((box.max.Y - box.min.Y) / cell))
    x0, y0 = box.min.X, box.min.Y
    yy, xx = np.mgrid[0:ny, 0:nx]
    centres = np.column_stack([(x0 + (xx.ravel() + 0.5) * cell),
                               (y0 + (yy.ravel() + 0.5) * cell)])
    solid = np.zeros(ny * nx, dtype=bool)
    for face in section(part, Plane.XY.offset(z)).faces():
        inside = Path(_outline(face.outer_wire(), cell / 2)).contains_points(centres)
        for hole in face.inner_wires():
            inside &= ~Path(_outline(hole, cell / 2)).contains_points(centres)
        solid |= inside
    return solid.reshape(ny, nx), x0, y0, box.min.Z, box.max.Z


def _disc(shape, x0, y0, cell, centre, radius):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cx = x0 + (xx + 0.5) * cell
    cy = y0 + (yy + 0.5) * cell
    return (cx - centre[0]) ** 2 + (cy - centre[1]) ** 2 <= radius ** 2


def _pockets(removed: np.ndarray, x0: float, y0: float, z0: float, z1: float,
             cell: float):
    """Prisms through the plate for the regions to cut, holes inside them
    (islands of material) added back by nesting depth."""
    polys = [p for p in topology.to_polygons(removed.astype(float), cell)
             if len(p) >= 4]
    paths = [Path(p) for p in polys]
    cutter = None
    for k, p in enumerate(polys):
        area = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1))
                         - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
        if area < MIN_POCKET_MM2:
            continue
        depth = sum(paths[m].contains_point(p[0]) for m in range(len(polys)) if m != k)
        pts = [Vector(x0 + x, y0 + y, z0 - 1.0) for x, y in p[:-1]]
        prism = extrude(Face(Wire.make_polygon(pts, close=True)), amount=z1 - z0 + 2.0)
        if depth % 2 == 0:
            cutter = prism if cutter is None else cutter + prism
        elif cutter is not None:
            cutter = cutter - prism
    return cutter


def hollow(part: Part, root_xy, tip_xy, boss_r: float, cell: float = CELL_MM,
           volfrac: float = VOLFRAC):
    """Hollowed copy of a flat part lying in XY, clamped at root_xy (the horn
    end) and loaded at tip_xy; returns (part, volume ratio)."""
    solid, x0, y0, z0, z1 = _grid(part, cell)
    rim = distance_transform_edt(solid) <= RIM_MM / cell
    root = _disc(solid.shape, x0, y0, cell, root_xy, boss_r) & solid
    tip = _disc(solid.shape, x0, y0, cell, tip_xy, boss_r) & solid
    keep = solid & (rim | root | tip)
    dx, dy = tip_xy[0] - root_xy[0], tip_xy[1] - root_xy[1]
    n = math.hypot(dx, dy) or 1.0
    loads = []
    for fx, fy in ((-dy / n, dx / n), (dx / n, dy / n)):   # across, then along
        f = np.zeros(solid.shape + (2,))
        f[tip] = (fx / tip.sum(), fy / tip.sum())
        loads.append(f)
    density = topology.simp(solid, keep, root, loads, volfrac=volfrac)
    removed = solid & ~keep & (density < THRESHOLD)
    cutter = _pockets(removed, x0, y0, z0, z1, cell)
    if cutter is None:
        return part, 1.0
    out = part - cutter
    return out, out.volume / part.volume

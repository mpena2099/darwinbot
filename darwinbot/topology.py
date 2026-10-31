"""
topology.py
-----------
SIMP topology optimisation for flat printed parts: a fixed grid of bilinear
4-node plane-stress elements, one density per cell as the design variable, a
sensitivity filter and the optimality-criteria update of the 88-line
formulation (Andreassen et al., 2011). A density field turns back into closed
polygons with contourpy for the CAD side.

Conventions: cell (i, j) covers x in [j * cell, (j + 1) * cell] and y in
[i * cell, (i + 1) * cell]; its four nodes are its corners, numbered
counter-clockwise from the bottom left. Lengths are in the caller's unit (mm on
the real parts) and E = 1, so only the shape and the load ratio matter.
"""

import math

import contourpy
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

VOID = 1e-3     # density of a cell outside the solid mask
NU = 0.3        # Poisson ratio
MOVE = 0.2      # largest density change per optimality-criteria update


def _element_stiffness() -> np.ndarray:
    """Element stiffness matrix of a unit square, plane stress, E = 1."""
    k = np.array([1 / 2 - NU / 6, 1 / 8 + NU / 8, -1 / 4 - NU / 12,
                  -1 / 8 + 3 * NU / 8, -1 / 4 + NU / 12, -1 / 8 - NU / 8,
                  NU / 6, 1 / 8 - 3 * NU / 8])
    order = [[0, 1, 2, 3, 4, 5, 6, 7],
             [1, 0, 7, 6, 5, 4, 3, 2],
             [2, 7, 0, 5, 6, 3, 4, 1],
             [3, 6, 5, 0, 7, 2, 1, 4],
             [4, 5, 6, 7, 0, 1, 2, 3],
             [5, 4, 3, 2, 1, 0, 7, 6],
             [6, 3, 4, 1, 2, 7, 0, 5],
             [7, 2, 1, 4, 3, 6, 5, 0]]
    return k[np.array(order)] / (1.0 - NU ** 2)


def _dofs(ny: int, nx: int) -> np.ndarray:
    """Dofs of every element, (ny * nx, 8), nodes counter-clockwise from the bottom left."""
    i, j = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    n = i * (nx + 1) + j                       # bottom left node of the cell
    nodes = np.stack((n, n + 1, n + nx + 2, n + nx + 1), axis=-1).reshape(-1, 4)
    return np.stack((2 * nodes, 2 * nodes + 1), axis=-1).reshape(-1, 8)


def _load_vector(dofs: np.ndarray, force: np.ndarray, ndof: int) -> np.ndarray:
    """Global force vector: a cell's (fx, fy) spread equally on its 4 nodes."""
    f = np.zeros(ndof)
    np.add.at(f, dofs[:, 0::2].ravel(), np.repeat(force[..., 0].ravel() / 4.0, 4))
    np.add.at(f, dofs[:, 1::2].ravel(), np.repeat(force[..., 1].ravel() / 4.0, 4))
    return f


def _stiffness_matrix(dofs: np.ndarray, density: np.ndarray, penal: float,
                      ndof: int) -> csc_matrix:
    """Global stiffness matrix, SIMP: E = density ** penal."""
    ke = _element_stiffness()
    stiff = density.ravel() ** penal
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    data = (stiff[:, None] * ke.ravel()[None, :]).ravel()
    return coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsc()


def _masks(solid: np.ndarray, fixed: np.ndarray, loads: list) -> tuple:
    """Coerce and validate the masks and the load cases."""
    solid = np.asarray(solid, dtype=bool)
    fixed = np.asarray(fixed, dtype=bool)
    if solid.ndim != 2:
        raise ValueError("the masks must be 2D")
    if fixed.shape != solid.shape:
        raise ValueError("fixed must have the shape of solid")
    if not loads:
        raise ValueError("at least one load case is required")
    loads = [np.asarray(force, dtype=float) for force in loads]
    for force in loads:
        if force.shape != solid.shape + (2,):
            raise ValueError("each load case must have shape (ny, nx, 2)")
    return solid, fixed, loads


def _solve(density: np.ndarray, fixed: np.ndarray, loads: list,
           penal: float) -> tuple:
    """Displacements and load vectors of every load case, on the free dofs."""
    ny, nx = density.shape
    dofs = _dofs(ny, nx)
    ndof = 2 * (ny + 1) * (nx + 1)
    clamped = np.unique(dofs[fixed.ravel()])
    free = np.setdiff1d(np.arange(ndof), clamped)
    k = _stiffness_matrix(dofs, density, penal, ndof)[free][:, free]
    out = []
    for force in loads:
        f = _load_vector(dofs, force, ndof)
        u = np.zeros(ndof)
        u[free] = spsolve(k, f[free])
        out.append((u, f))
    return dofs, out


def _filter(solid: np.ndarray, rmin: float) -> tuple:
    """Sensitivity filter over the solid cells: H and its row sums."""
    ny, nx = solid.shape
    n = ny * nx
    index = np.arange(n).reshape(ny, nx)
    flat = solid.ravel()
    radius = int(math.ceil(rmin))
    rows, cols, vals = [], [], []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            w = rmin - math.hypot(di, dj)
            i0, i1 = max(0, -di), min(ny, ny - di)
            j0, j1 = max(0, -dj), min(nx, nx - dj)
            if w <= 0.0 or i0 >= i1 or j0 >= j1:
                continue
            dst = index[i0:i1, j0:j1]
            src = index[i0 + di:i1 + di, j0 + dj:j1 + dj]
            keep = flat[dst] & flat[src]
            rows.append(dst[keep])
            cols.append(src[keep])
            vals.append(np.full(keep.sum(), w))
    if not vals:                               # rmin <= 0: no filtering
        return coo_matrix((n, n)).tocsr(), np.ones(n)
    h = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()
    return h, np.maximum(np.asarray(h.sum(axis=1)).ravel(), 1.0)


def _oc(x: np.ndarray, dc: np.ndarray, design: np.ndarray,
        target: float) -> np.ndarray:
    """One optimality-criteria update of the design cells for a volume target."""
    cell = design.ravel()
    current = x.ravel()
    trial = current
    lo, hi = 1e-9, 1e9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        trial = np.clip(current * np.sqrt(np.maximum(-dc, 0.0) / mid),
                        np.maximum(VOID, current - MOVE),
                        np.minimum(1.0, current + MOVE))
        if trial[cell].sum() > target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) / (lo + hi) < 1e-5:
            break
    out = x.copy()
    out.ravel()[cell] = trial[cell]
    return out


def compliance(density: np.ndarray, solid: np.ndarray, fixed: np.ndarray,
               loads: list) -> float:
    """Sum of the compliances of every load case for a density field."""
    density = np.asarray(density, dtype=float)
    solid, fixed, loads = _masks(solid, fixed, loads)
    if density.shape != solid.shape:
        raise ValueError("density must have the shape of solid")
    _, solved = _solve(density, fixed, loads, 3.0)  # default exponent of simp
    return sum(float(f @ u) for u, f in solved)


def simp(solid: np.ndarray, keep: np.ndarray, fixed: np.ndarray, loads: list,
         volfrac: float = 0.4, rmin: float = 1.5,
         penal: float = 3.0, iters: int = 60) -> np.ndarray:
    """
    Optimise the density of every cell of `solid` for the sum of the
    compliances of the load cases. Cells in `keep` stay at 1, cells outside
    `solid` stay at VOID and the four nodes of `fixed` cells are clamped.
    `volfrac` is the fraction of the solid area kept, counting the keep cells.
    Returns densities in [0, 1].
    """
    solid, fixed, loads = _masks(solid, fixed, loads)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != solid.shape:
        raise ValueError("keep must have the shape of solid")
    if not 0.0 < volfrac <= 1.0:
        raise ValueError("volfrac must be in (0, 1]")
    ny, nx = solid.shape
    x = np.where(solid, volfrac, VOID)
    x[keep] = 1.0
    design = solid & ~keep
    target = volfrac * float(solid.sum()) - float(keep.sum())
    h, hs = _filter(solid, rmin)
    ke = _element_stiffness()
    for _ in range(iters):
        dofs, solved = _solve(x, fixed, loads, penal)
        dc = np.zeros(ny * nx)
        for u, _ in solved:
            ue = u[dofs]
            dc -= penal * x.ravel() ** (penal - 1.0) * np.einsum(
                "ij,jk,ik->i", ue, ke, ue)
        dc = (h @ (x.ravel() * dc)) / hs / np.maximum(VOID, x.ravel())
        x = _oc(x, dc, design, target)
    return x


def to_polygons(density: np.ndarray, cell: float,
                threshold: float = 0.5) -> list:
    """
    Closed polygons of the region where `density` is at least `threshold`.
    Outer boundaries and holes are both returned, in contourpy's order, with x
    and y in the same unit as `cell`.
    """
    density = np.asarray(density, dtype=float)
    field = np.zeros((density.shape[0] + 2, density.shape[1] + 2))
    field[1:-1, 1:-1] = density
    generator = contourpy.contour_generator(z=field)
    polygons = []
    for line in generator.lines(threshold):
        if len(line) < 3:
            continue
        polygons.append(np.column_stack(((line[:, 0] - 0.5) * cell,
                                         (line[:, 1] - 0.5) * cell)))
    return polygons

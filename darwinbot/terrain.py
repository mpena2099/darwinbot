"""
terrain.py
----------
Terrains the robot is evaluated on. Each one is a self-contained MJCF
snippet (assets + worldbody geoms) that includes its own ground.

Terrain geoms carry priority="1" so their friction overrides the robot's.
Without it MuJoCo takes the maximum of the two frictions, and a low-friction
floor would never feel slippery.

Features start close to the spawn point (x = 0): a typical evolved gait
covers 1 to 3 m per episode, so anything farther out would rarely be reached.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

GROUND = 'priority="1" condim="3"'


@dataclass(frozen=True)
class Terrain:
    name:   str
    assets: str = ""
    world:  str = ""


def _plane(friction: float = 1.0) -> str:
    return (f'<geom name="floor" type="plane" size="30 30 0.1" material="grid" '
            f'friction="{friction} 0.02 0.001" {GROUND}/>')


def flat() -> Terrain:
    return Terrain("flat", world=_plane())


def rough(height: float = 0.04, seed: int = 7) -> Terrain:
    """Smoothed random bumps; flat around the spawn so the robot lands cleanly."""
    nrow, ncol = 60, 200                # y, x samples
    half_x, half_y, cx = 6.0, 2.0, 5.0  # field covers x in [-1, 11]
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1, (nrow, ncol))
    kernel = np.ones(5) / 5
    for _ in range(3):
        z = np.apply_along_axis(np.convolve, 1, z, kernel, mode="same")
        z = np.apply_along_axis(np.convolve, 0, z, kernel, mode="same")
    z = (z - z.min()) / (z.max() - z.min())
    xs = np.linspace(cx - half_x, cx + half_x, ncol)
    z[:, xs < 0.6] = 0.0
    elevation = " ".join(f"{v:.3f}" for v in z.ravel())
    return Terrain(
        "rough",
        assets=(f'<hfield name="bumps" nrow="{nrow}" ncol="{ncol}" '
                f'size="{half_x} {half_y} {height} 0.05" elevation="{elevation}"/>'),
        world=(_plane() +
               f'<geom type="hfield" hfield="bumps" pos="{cx} 0 0" '
               f'rgba="0.55 0.50 0.42 1" friction="1.0 0.02 0.001" {GROUND}/>'),
    )


def steps(rise: float = 0.03, run: float = 0.5, count: int = 6) -> Terrain:
    """Ascending staircase, first riser at x = 0.6."""
    geoms = [_plane()]
    for i in range(count):
        x0 = 0.6 + i * run
        x1 = 0.6 + count * run + 2.0    # each step extends past the top
        h  = rise * (i + 1)
        geoms.append(
            f'<geom type="box" pos="{(x0 + x1) / 2:.3f} 0 {h / 2:.3f}" '
            f'size="{(x1 - x0) / 2:.3f} 2 {h / 2:.3f}" '
            f'rgba="0.75 0.65 0.55 1" friction="1.0 0.02 0.001" {GROUND}/>')
    return Terrain("steps", world="".join(geoms))


def step(height: float = 0.03, start: float = 0.30, depth: float = 1.0) -> Terrain:
    """One box step across the path, `start` metres ahead of the spawn."""
    return Terrain("step", world=_plane() + (
        f'<geom type="box" pos="{start + depth / 2:.3f} 0 {height / 2:.3f}" '
        f'size="{depth / 2:.3f} 2 {height / 2:.3f}" '
        f'rgba="0.75 0.65 0.55 1" friction="1.0 0.02 0.001" {GROUND}/>'))


def slope(degrees: float = 8.0, start: float = 0.5, length: float = 6.0) -> Terrain:
    a = np.radians(degrees)
    h = 0.05                                      # half thickness
    along  = np.array([np.cos(a), 0, np.sin(a)])
    normal = np.array([-np.sin(a), 0, np.cos(a)])
    centre = np.array([start, 0, 0]) + along * length / 2 - normal * h
    return Terrain("slope", world=_plane() + (
        f'<geom type="box" pos="{centre[0]:.4f} 0 {centre[2]:.4f}" '
        f'size="{length / 2} 2 {h}" euler="0 {-a:.4f} 0" '
        f'rgba="0.60 0.75 0.60 1" friction="1.0 0.02 0.001" {GROUND}/>'))


def slippery(friction: float = 0.2) -> Terrain:
    """Low-friction floor everywhere: tests gaits that rely on grip."""
    return Terrain("slippery", world=_plane(friction))


TERRAINS = {
    "flat": flat, "rough": rough, "steps": steps, "step": step,
    "slope": slope, "slippery": slippery,
}

PRESETS: Dict[str, List[str]] = {
    "flat":    ["flat"],
    "terrain": ["flat", "rough", "steps"],
    "all":     ["flat", "rough", "steps", "slope", "slippery"],
}


def get_terrains(spec: str) -> List[Terrain]:
    names = PRESETS.get(spec, [n.strip() for n in spec.split(",")])
    unknown = [n for n in names if n not in TERRAINS]
    if unknown:
        raise ValueError(f"Unknown terrain(s) {unknown}. "
                         f"Available: {list(TERRAINS)}; presets: {list(PRESETS)}")
    return [TERRAINS[n]() for n in names]

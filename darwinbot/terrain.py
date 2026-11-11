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

Random terrains (rough, rubble) take a seed. Training draws a new one for every
body, so a lineage cannot memorise one set of bumps; replay and validation use
the fixed default, so the same robot always sees the same ground there.
"""

import inspect
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

GROUND = 'priority="1" condim="3"'


@dataclass(frozen=True)
class Terrain:
    name:   str
    assets: str = ""
    world:  str = ""
    friction: float = 1.0   # ground friction; skids and spikes scale with it
    spawn_z:  float = 0.0   # ground height under the spawn point


def _plane(friction: float = 1.0, extra: str = "") -> str:
    return (f'<geom name="floor" type="plane" size="30 30 0.1" material="grid" '
            f'friction="{friction} 0.02 0.001" {GROUND} {extra}/>')


def _box(x: float, y: float, z: float, sx: float, sy: float, sz: float,
         rgba: str = "0.75 0.65 0.55 1", yaw: float = 0.0) -> str:
    """Box ground geom from its centre and half sizes (m), yaw in radians."""
    return (f'<geom type="box" pos="{x:.3f} {y:.3f} {z:.3f}" '
            f'size="{sx:.3f} {sy:.3f} {sz:.3f}" euler="0 0 {yaw:.3f}" '
            f'rgba="{rgba}" friction="1.0 0.02 0.001" {GROUND}/>')


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


def step(height: float = 0.03, start: float = 0.50, depth: float = 1.0) -> Terrain:
    """One box step across the path, `start` metres ahead of the spawn (far
    enough that long front limbs do not start embedded in it)."""
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
    return Terrain("slippery", world=_plane(friction), friction=friction)


def rubble(seed: int = 11, count: int = 40, low: float = 0.03,
           high: float = 0.07) -> Terrain:
    """Scattered rocks 3 to 7 cm high, taller than many wheels can climb: a
    field that rewards stepping over things instead of rolling into them."""
    rng = np.random.default_rng(seed)
    geoms = [_plane()]
    for _ in range(count):
        h = rng.uniform(low, high)
        geoms.append(_box(rng.uniform(0.6, 4.0), rng.uniform(-0.6, 0.6), h / 2,
                          rng.uniform(0.02, 0.06), rng.uniform(0.02, 0.06), h / 2,
                          "0.55 0.52 0.50 1", rng.uniform(0, np.pi)))
    return Terrain("rubble", world="".join(geoms))


def ledges(rise: float = 0.06, run: float = 0.6, count: int = 4) -> Terrain:
    """Staircase of 6 cm steps, twice the old ones: too tall to roll up."""
    return Terrain("ledges", world=steps(rise, run, count).world)


def soft(friction: float = 1.0) -> Terrain:
    """Yielding ground like sand or carpet: soft contacts let feet sink one or two
    centimetres and rolling friction (condim 6) drags on wheels."""
    return Terrain("soft", friction=friction, world=(
        f'<geom name="floor" type="plane" size="30 30 0.1" material="grid" '
        f'friction="{friction} 0.02 0.02" priority="1" condim="6" '
        f'solref="0.08 1" solimp="0.2 0.9 0.03"/>'))


def downhill(degrees: float = 10.0, start: float = 0.5, length: float = 1.5) -> Terrain:
    """Spawn on a plateau, then a slope down to the floor: descending without
    tumbling. The plateau is as high as the slope drops."""
    a = np.radians(degrees)
    top = length * np.sin(a)
    h = 0.05
    along = np.array([np.cos(a), 0, -np.sin(a)])
    normal = np.array([np.sin(a), 0, np.cos(a)])
    centre = np.array([start, 0, top]) + along * length / 2 - normal * h
    plateau = _box(start - 1.5, 0, top / 2, 1.5, 2, top / 2, "0.60 0.75 0.60 1")
    return Terrain("downhill", spawn_z=float(top), world=_plane() + plateau + (
        f'<geom type="box" pos="{centre[0]:.4f} 0 {centre[2]:.4f}" '
        f'size="{length / 2} 2 {h}" euler="0 {a:.4f} 0" '
        f'rgba="0.60 0.75 0.60 1" friction="1.0 0.02 0.001" {GROUND}/>'))


def steep(degrees: float = 15.0) -> Terrain:
    """The slope at 15 degrees instead of 8."""
    return Terrain("steep", world=slope(degrees).world)


TERRAINS = {
    "flat": flat, "rough": rough, "steps": steps, "step": step,
    "slope": slope, "slippery": slippery, "rubble": rubble, "ledges": ledges,
    "soft": soft, "downhill": downhill, "steep": steep,
}
SEEDED = ["rough", "rubble"]    # terrains drawn anew for every body in training


def register_terrain(name: str, factory, seeded: bool = False):
    """Register an external terrain, loaded by a run's --plugin file. A built-in
    name cannot be replaced. `seeded` terrains take a seed in training."""
    if name in TERRAINS:
        raise ValueError(f"cannot overwrite terrain '{name}'")
    TERRAINS[name] = factory
    if seeded:
        SEEDED.append(name)         # in place: other modules import the list
    return factory


PRESETS: Dict[str, List[str]] = {
    "flat":    ["flat"],
    "terrain": ["flat", "rough", "steps", "rubble"],
    "all":     ["flat", "rough", "steps", "rubble", "ledges", "slope", "steep",
                "downhill", "slippery", "soft"],
}


def parse_spec(spec: str) -> List[Tuple[str, Dict[str, float]]]:
    """(name, keyword arguments) of every terrain in a preset name or a comma
    list, where each item may set its factory's arguments after colons:
    "steps:rise=0.08:run=0.3,slope:degrees=12". Unknown terrains, unknown
    arguments and values that are not numbers are refused."""
    items = PRESETS.get(spec, [n.strip() for n in spec.split(",")])
    out = []
    for item in items:
        name, *args = [x.strip() for x in item.split(":")]
        if name not in TERRAINS:
            raise ValueError(f"Unknown terrain {name!r}. "
                             f"Available: {list(TERRAINS)}; presets: {list(PRESETS)}")
        params = inspect.signature(TERRAINS[name]).parameters
        kwargs = {}
        for arg in args:
            key, eq, value = arg.partition("=")
            key = key.strip()
            if not eq or key not in params or key == "seed":
                allowed = [k for k in params if k != "seed"]
                raise ValueError(f"terrain {name!r}: bad argument {arg!r}, "
                                 f"expected key=value with key in {allowed}")
            try:
                v = float(value)
            except ValueError:
                raise ValueError(f"terrain {name!r}: {key}={value!r} is not a number")
            if isinstance(params[key].default, int):
                if v != int(v):
                    raise ValueError(f"terrain {name!r}: {key} must be a whole number")
                v = int(v)
            kwargs[key] = v
        out.append((name, kwargs))
    return out


def get_terrains(spec: str, seed: Optional[int] = None) -> List[Terrain]:
    """Terrains from a preset name or a comma list (see parse_spec). With a
    seed, the random terrains (SEEDED) are drawn from it; without, they use
    their fixed default. A terrain built with arguments is named after its
    item, e.g. "steps:rise=0.08"."""
    out = []
    for k, (n, kwargs) in enumerate(parse_spec(spec)):
        if seed is not None and n in SEEDED:
            t = TERRAINS[n](seed=(seed + 7919 * k) % 2**31, **kwargs)
        else:
            t = TERRAINS[n](**kwargs)
        if kwargs:
            t = replace(t, name=":".join([n] + [f"{a}={v:g}" for a, v in kwargs.items()]))
        out.append(t)
    return out

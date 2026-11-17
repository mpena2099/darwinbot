"""
robot_file.py
-------------
A robot written by hand in a YAML file, and the other way round: an evolved
robot written out as YAML so a person can edit it. See docs/own-robot.md.

  name: my-car
  torso: {length_mm: 160, width_mm: 120, contour: [1,1,1,1,1,1,1,1], lock: true}
  nodes:
    left_wheel: {parent: torso, slot: 2, kind: wheel, radius_mm: 60, lock: true}
    foot:       {parent: hip, slot: 0, kind: foot, radius_mm: 20, shape: pad}

Units are millimetres and degrees; attachment fractions stay 0..1. `load`
checks every value against the genome ranges and then that decode(encode(body))
gives the same body back, so what the file says is what evolution starts from.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from darwinbot import morphology as mo

LOCK = "lock"
NODE_KEYS_ALL = ("parent", "slot", "kind", "id", LOCK, "roll_deg", "offset", "angle")
TOP_KEYS = ("name", "torso", "nodes")
TORSO_KEYS = ("length_mm", "width_mm", "contour", LOCK)
TOL = 1e-5


class _F:
    """One YAML field of a node: its Node attribute, conversion and range."""

    def __init__(self, attr, scale=1.0, rng=None, choices=None, kind="num"):
        self.attr, self.scale, self.rng, self.choices, self.kind = (
            attr, scale, rng, choices, kind)


def _mm(attr, rng):
    return _F(attr, 0.001, rng)


_LINK = {
    "length_mm": _mm("length", mo.LENGTH_RANGE),
    "width_mm": _mm("width", mo.LINK_WIDTH_RANGE),
    "thick_mm": _mm("thick", mo.LINK_THICK_RANGE),
    "profile": _F("profile", choices=mo.PROFILES, kind="choice"),
    "bend_deg": _F("bend"),                    # degrees in the file, radians in Node
}
_AXIS = {"axis": _F("axis", choices=mo.AXES, kind="choice"),
         "flip": _F("flip", kind="bool")}

FIELDS: Dict[str, Dict[str, _F]] = {
    "joint": {**_AXIS, **_LINK},
    "link": dict(_LINK),
    "spring": {"flip": _AXIS["flip"], **_LINK,
               "thick_mm": _mm("thick", mo.SPRING_BLADE_RANGE)},
    "segment": {**_AXIS,
                "length_mm": _mm("length", mo.SEGMENT_LENGTH_RANGE),
                "width_mm": _mm("width", mo.SEGMENT_WIDTH_RANGE)},
    "wheel": {"radius_mm": _mm("wheel_rad", mo.WHEEL_RANGE_M),
              "width_mm": _mm("wheel_width", mo.WHEEL_WIDTH_RANGE)},
    "caster": {"radius_mm": _mm("wheel_rad", mo.CASTER_RANGE)},
    "foot": {"radius_mm": _mm("foot_rad", mo.FOOT_RANGE),
             "shape": _F("foot_shape", choices=mo.FOOT_SHAPES, kind="choice")},
    "tail": {"length_mm": _mm("length", mo.LENGTH_RANGE),
             "tail_mass_g": _F("tail_mass", 0.001, mo.TAIL_MASS_RANGE)},
    "skid": {},
    "spike": {},
}

# Node fields that a round trip must reproduce, and the YAML key they came from.
_COMPARE = ("kind", "parent", "slot", "rot", "axis", "length", "wheel_rad",
            "wheel_width", "flip", "width", "thick", "foot_rad", "foot_shape",
            "tail_mass", "offset", "angle", "profile", "bend")


_COMMON = ("kind", "parent", "slot", "rot", "offset", "angle")


def _yaml_key(kind: str, attr: str) -> str:
    if attr == "rot":
        return "roll_deg"
    for key, f in FIELDS.get(kind, {}).items():
        if f.attr == attr:
            return key
    return attr


def _num(where: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"{where}: {value!r} is not a finite number")
    return float(value)


def _in_range(where: str, v: float, lo: float, hi: float, unit: str = ""):
    if v < lo - 1e-9 or v > hi + 1e-9:
        raise ValueError(f"{where}: {v:g}{unit} is outside the allowed range "
                         f"{lo:g} to {hi:g}{unit}")


def _check_keys(where: str, given, allowed):
    for key in given:
        if key not in allowed:
            raise ValueError(f"{where}: unknown key '{key}', expected one of "
                             f"{', '.join(allowed)}")


def _node(name: str, spec: dict, kind: str) -> mo.Node:
    fields = FIELDS[kind]
    _check_keys(f"node '{name}' ({kind})", spec, NODE_KEYS_ALL + tuple(fields))
    node = mo.Node(0, kind)
    if kind == "spring":
        node.axis = "yaw"                      # a spring bends about its blade
    for key, f in fields.items():
        if key not in spec:
            if f.rng is not None and f.kind == "num" and key != "bend_deg":
                # an omitted value takes the default, moved into the range
                setattr(node, f.attr, min(max(getattr(node, f.attr), f.rng[0]), f.rng[1]))
            continue
        where = f"node '{name}' {key}"
        value = spec[key]
        if f.kind == "choice":
            if value not in f.choices:
                raise ValueError(f"{where}: '{value}' is not one of "
                                 f"{', '.join(f.choices)}")
            setattr(node, f.attr, value)
        elif f.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{where}: expected true or false, got {value!r}")
            setattr(node, f.attr, value)
        else:
            v = _num(where, value)
            if key == "bend_deg":
                _in_range(where, v, *mo.BEND_RANGE, unit=" deg")
                setattr(node, f.attr, float(np.radians(v)))
            else:
                scale = 1000.0
                lo, hi = f.rng[0] * scale, f.rng[1] * scale
                unit = " g" if key.endswith("_g") else " mm"
                _in_range(where, v, lo, hi, unit)
                setattr(node, f.attr, v / scale)
    for key in ("offset", "angle"):
        if key in spec:
            v = _num(f"node '{name}' {key}", spec[key])
            _in_range(f"node '{name}' {key}", v, 0.0, 1.0)
            setattr(node, key, v)
    if "roll_deg" in spec:
        v = _num(f"node '{name}' roll_deg", spec["roll_deg"])
        node.rot = float(np.radians(v % 360.0))
    return node


def load(path) -> Tuple[mo.Morphology, Tuple[int, ...]]:
    """Read a robot file: (body, locked node ids; 0 is the torso)."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: not valid YAML: {e}") from e
    return parse(doc)


def parse(doc) -> Tuple[mo.Morphology, Tuple[int, ...]]:
    if not isinstance(doc, dict):
        raise ValueError("a robot file is a mapping with torso and nodes")
    _check_keys("robot file", doc, TOP_KEYS)
    torso = doc.get("torso") or {}
    if not isinstance(torso, dict):
        raise ValueError("torso: expected a mapping")
    _check_keys("torso", torso, TORSO_KEYS)
    morph = mo.Morphology(0.25, 0.15, {0: mo.Node(0, "core")})
    for key, bounds, attr in (("length_mm", mo.BODY_BOUNDS[0], "body_length"),
                              ("width_mm", mo.BODY_BOUNDS[1], "body_width")):
        if key in torso:
            v = _num(f"torso {key}", torso[key])
            _in_range(f"torso {key}", v, bounds[0] * 1000, bounds[1] * 1000, " mm")
            setattr(morph, attr, v / 1000.0)
    if "contour" in torso:
        c = torso["contour"]
        if not isinstance(c, (list, tuple)) or len(c) != 8:
            raise ValueError("torso contour: expected a list of 8 factors")
        for k, v in enumerate(c):
            v = _num(f"torso contour[{k}]", v)
            _in_range(f"torso contour[{k}]", v, *mo.CONTOUR_RANGE)
        morph.contour = tuple(float(v) for v in c)
    locked: List[int] = []
    if _flag("torso", torso):
        locked.append(0)

    nodes = doc.get("nodes") or {}
    if not isinstance(nodes, dict):
        raise ValueError("nodes: expected a mapping of name to node")
    if len(nodes) + 1 > mo.MAX_NODES:
        raise ValueError(f"{len(nodes)} nodes plus the torso is more than "
                         f"MAX_NODES = {mo.MAX_NODES}")
    ids: Dict[str, int] = {"torso": 0}
    explicit = set()
    for name, spec in nodes.items():
        if name == "torso":
            raise ValueError("a node cannot be called 'torso'")
        if isinstance(spec, dict) and "id" in spec:
            i = spec["id"]
            if (not isinstance(i, int) or isinstance(i, bool)
                    or not 1 <= i < mo.MAX_NODES):
                raise ValueError(f"node '{name}' id: expected an integer in "
                                 f"1..{mo.MAX_NODES - 1}")
            if i in explicit:
                raise ValueError(f"node '{name}' id: {i} is used twice")
            explicit.add(i)
    free = (i for i in range(1, mo.MAX_NODES) if i not in explicit)
    taken: Dict[Tuple[int, int], str] = {}
    for name, spec in nodes.items():
        if not isinstance(spec, dict):
            raise ValueError(f"node '{name}': expected a mapping")
        for key in ("parent", "slot", "kind"):
            if key not in spec:
                raise ValueError(f"node '{name}': missing '{key}'")
        kind = spec["kind"]
        if kind not in FIELDS:
            raise ValueError(f"node '{name}' kind: unknown '{kind}', expected one "
                             f"of {', '.join(FIELDS)}")
        parent = spec["parent"]
        if parent not in ids:
            raise ValueError(f"node '{name}' parent: '{parent}' is not 'torso' or "
                             f"a node listed before it")
        node = _node(name, spec, kind)
        node.id = spec["id"] if "id" in spec else next(free)
        node.parent = ids[parent]
        slot = spec["slot"]
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise ValueError(f"node '{name}' slot: expected an integer, got {slot!r}")
        pkind = morph.nodes[node.parent].kind
        if mo.N_SLOTS[pkind] == 0:
            raise ValueError(f"node '{name}' parent: a {pkind} has no slots")
        if not 0 <= slot < mo.N_SLOTS[pkind]:
            raise ValueError(f"node '{name}' slot: {slot} is outside 0 to "
                             f"{mo.N_SLOTS[pkind] - 1} for a {pkind}")
        if pkind == "segment" and slot == 1:
            raise ValueError(f"node '{name}' slot: slot 1 of a segment faces its parent")
        if kind == "segment" and not mo._segment_slot(pkind, slot):
            raise ValueError(f"node '{name}' slot: a segment attaches only to slot "
                             f"0 or 1 of the torso or of another segment")
        if (node.parent, slot) in taken:
            raise ValueError(f"node '{name}' slot: slot {slot} of '{parent}' is "
                             f"already used by '{taken[(node.parent, slot)]}'")
        taken[(node.parent, slot)] = name
        node.slot = slot
        morph.nodes[node.id] = node
        ids[name] = node.id
        if _flag(f"node '{name}'", spec):
            locked.append(node.id)
    _check_round_trip(morph, ids)
    return morph, tuple(sorted(locked))


def _flag(where: str, spec: dict) -> bool:
    v = spec.get(LOCK, False)
    if not isinstance(v, bool):
        raise ValueError(f"{where} {LOCK}: expected true or false, got {v!r}")
    return v


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= TOL
    return a == b


def _check_round_trip(morph: mo.Morphology, ids: Dict[str, int]):
    """decode(encode(morph)) must be the same body: otherwise a value fell out of
    the genome (a limit such as depth or servo count) and evolution would start
    from a different robot than the file describes."""
    back = mo.decode(mo.encode(morph))
    names = {i: n for n, i in ids.items()}
    for i, n in morph.nodes.items():
        if i == 0:
            continue
        name = names.get(i, f"#{i}")
        m = back.nodes.get(i)
        if m is None:
            raise ValueError(
                f"node '{name}': dropped by the genome. A body holds at most "
                f"{mo.MAX_SERVOS} servos, {mo.MAX_SEGMENTS} segments and depth "
                f"{mo.MAX_DEPTH} (a segment does not count)")
        used = {f.attr for f in FIELDS[n.kind].values()}
        for attr in _COMPARE:
            if attr not in _COMMON and attr not in used and not (
                    attr == "axis" and n.kind == "spring"):
                continue                    # a gene this kind does not read
            if not _same(getattr(n, attr), getattr(m, attr)):
                key = _yaml_key(n.kind, attr)
                raise ValueError(
                    f"node '{name}' {key}: {getattr(n, attr)!r} is not kept by the "
                    f"genome (it becomes {getattr(m, attr)!r})")
    for attr in ("body_length", "body_width"):
        if not _same(getattr(morph, attr), getattr(back, attr)):
            raise ValueError(f"torso {attr}: not kept by the genome")
    if not all(_same(a, b) for a, b in zip(morph.contour, back.contour)):
        raise ValueError("torso contour: not kept by the genome")


# --------------------------------------------------------------------------
# dump
# --------------------------------------------------------------------------

def _round(v: float, digits: int = 4) -> float:
    r = round(float(v), digits)
    return int(r) if r == int(r) else r


def _flow(d: dict) -> str:
    return yaml.safe_dump(d, default_flow_style=True, width=10**6,
                          sort_keys=False).strip()


def dump(morph: mo.Morphology, locked=(), name: Optional[str] = None) -> str:
    """The body as robot YAML; node ids are written so a reload keeps them."""
    locked = set(locked)
    lines = []
    if name:
        lines.append(f"name: {name}")
    torso = {"length_mm": _round(morph.body_length * 1000),
             "width_mm": _round(morph.body_width * 1000),
             "contour": [_round(c, 6) for c in morph.contour]}
    if 0 in locked:
        torso[LOCK] = True
    lines.append(f"torso: {_flow(torso)}")
    lines.append("nodes:")
    names = {0: "torso"}
    done = {0}
    todo = sorted(i for i in morph.nodes if i != 0)
    while todo:
        ready = [i for i in todo if morph.nodes[i].parent in done]
        if not ready:
            raise ValueError("a node has a parent that is not in the body")
        i = ready[0]
        todo.remove(i)
        done.add(i)
        n = morph.nodes[i]
        names[i] = f"{n.kind}_{i}"
        spec = {"id": i, "parent": names[n.parent], "slot": n.slot, "kind": n.kind}
        for key, f in FIELDS[n.kind].items():
            v = getattr(n, f.attr)
            if f.kind == "num":
                v = (_round(np.degrees(v), 6) if key == "bend_deg"
                     else _round(v / f.scale, 3))
            spec[key] = v
        spec["roll_deg"] = _round(np.degrees(n.rot), 6)
        spec["offset"] = _round(n.offset, 6)
        spec["angle"] = _round(n.angle, 6)
        if i in locked:
            spec[LOCK] = True
        lines.append(f"  {names[i]}: {_flow(spec)}")
    return "\n".join(lines) + "\n"

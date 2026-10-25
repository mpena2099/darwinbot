"""
morphology.py
-------------
Body genome: a tree of real modules (Karl Sims / Revolve2 style).

Modules
  core   root, torso plate + ESP32 + LiPo. 8 slots: 0 front, 1 back,
         2..4 left side (front, middle, rear), 5..7 right side
  joint  one STS3215 servo + a printed link. Slots: 0 tip, 1 left, 2 right
         (the side slots are an elbow at the end of the link)
  link   passive printed link, same slots as the joint
  spring passive TPU flexure: a passive hinge plus a printed link, same slots
         as a link
  wheel  servo in wheel mode + wheel, terminal
  foot   TPU tip, terminal. A joint or link with no child ends in an implicit foot.
  tail   printed bar with a counterweight at the tip, terminal
  skid   low-friction pad, terminal
  spike  high-friction TPU tip, terminal
  caster free wheel on a passive hinge, terminal

Every slot also has a continuous roll in [0, 2*pi) around the slot direction,
so a child can point anywhere. A joint turns about its pitch (local Z), yaw
(local Y) or roll (local X) axis; local X is the direction of the link. `flip`
negates the axis, which is how a mirrored copy keeps "same signal, mirrored
motion".

Genome: MAX_NODES blocks of BLOCK_GENES genes in [0, 1], one per node slot.
The block index is the stable slot id of the node (the controller is indexed
by it). Block 0 is always the core. A block is
  [ACTIVE, PARENT, SLOT, ROT, TYPE, AXIS, LENGTH, WHEEL, FLIP,
   WIDTH, THICK, OFFSET, ANGLE]
(for the core, LENGTH and WHEEL hold body length and width). LENGTH is the
link length (40..200 mm); WIDTH and THICK are the link section (12..30 mm by
3..10 mm, and for a spring THICK is the TPU blade, 2..8 mm); WHEEL is the wheel
radius or, for a foot, the foot radius; ROT is
the slot roll in [0, 2*pi). OFFSET and ANGLE are attachment fractions in
[0, 1] whose meaning depends on the parent:
  * core front/back: OFFSET slides the mount across the edge (y) and ANGLE
    yaws it by up to +/-60 degrees;
  * core sides: OFFSET slides the mount along the edge (x) by +/-30% of half
    the body length and ANGLE yaws it the same way;
  * joint/link tip: OFFSET and ANGLE are ignored;
  * joint/link sides (elbow): OFFSET sets the elbow distance (25..80 mm) and
    ANGLE the elbow angle (30..150 degrees), so 0.5 is a 90 degree elbow.

decode() accepts any vector and always returns a body that respects
MAX_NODES, MAX_SERVOS and MAX_DEPTH: blocks that are inactive, unreachable,
clash on a slot or break a limit are dropped. Mutation works on the tree
(decode, edit, encode), so a repaired genome is just encode(decode(genes)).

Frames: X forward, Y left, Z up in the torso frame.
"""

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

MAX_NODES  = 20
MAX_SERVOS = 16
MAX_DEPTH  = 5      # core is depth 0

LENGTH_RANGE = (0.04, 0.20)               # m, joint / link length
LINK_WIDTH_RANGE = (0.012, 0.030)         # m, link width
LINK_THICK_RANGE = (0.003, 0.010)         # m, link thickness
SPRING_BLADE_RANGE = (0.002, 0.008)       # m, TPU blade thickness of a spring
FOOT_RANGE = (0.008, 0.03)                # m, TPU foot radius
TAIL_MASS_RANGE = (0.0, 0.10)             # kg, counterweight at the tip of a tail
TAIL_TIP_RADIUS = 0.012                   # m
TAIL_BAR_SIZE = (0.015, 0.006)            # m, width and thickness of the tail bar
SKID_SIZE = (0.06, 0.03, 0.006)           # m, low-friction pad
SKID_RADIUS = 0.015                       # m, contact radius of a skid
SPIKE_LENGTH = 0.03                       # m
SPIKE_RADIUS = 0.005                      # m
ELBOW_RANGE = (0.025, 0.08)               # m, side slot offset from the link axis
WHEEL_RANGE_M = (0.03, 0.06)              # m, wheel radius
CASTER_RANGE = (0.015, 0.035)             # m, free caster wheel radius
CASTER_FORK_SIZE = (0.02, 0.02, 0.02)     # m, printed fork box of a caster
CASTER_WHEEL_T = 0.008                    # m, caster wheel thickness
BODY_BOUNDS = [
    (0.15, 0.40),   # body_length (m)
    (0.10, 0.25),   # body_width  (m)
]
TORSO_HEIGHT = 0.06                       # m
SERVO_BOX = (0.045, 0.025, 0.035)   # m, length along the link, width, height
SIDE_X = (0.7, 0.0, -0.7)           # core side slots, fraction of half length
ELBOW_OFFSET = (0.04 - ELBOW_RANGE[0]) / (ELBOW_RANGE[1] - ELBOW_RANGE[0])

# Printability (see docs/plano-forma-livre.md, "Restricoes de construcao").
SERVO_STALL = 1.91       # N*m, STS3215 stall torque
PLA_STRENGTH = 50e6      # Pa, printed PLA
STRENGTH_SAFETY = 0.5    # link bending stress must stay below half of PLA
BED = 0.22               # m, largest part that fits a common print bed

# Spring (see docs/hardware-dados.md, "Mola de TPU"): a TPU blade bending at
# one end. E is not verified yet.
E_TPU = 26e6             # Pa, printed TPU
SPRING_FLEX_LEN = 0.012  # m, free blade length
SPRING_DAMPING = 0.02    # N*m*s/rad

TYPE_KINDS = ("joint", "link", "wheel", "foot", "spring",
              "tail", "skid", "spike", "caster")             # from the TYPE gene
AXES = ("pitch", "yaw", "roll")
AXIS_LOCAL = {"pitch": (0.0, 0.0, 1.0), "yaw": (0.0, 1.0, 0.0),
              "roll": (1.0, 0.0, 0.0)}

(ACTIVE, PARENT, SLOT, ROT, TYPE, AXIS, LENGTH, WHEEL, FLIP,
 WIDTH, THICK, OFFSET, ANGLE) = range(13)
BLOCK_GENES  = 13
MORPH_LENGTH = MAX_NODES * BLOCK_GENES

N_SLOTS = {"core": 8, "joint": 3, "link": 3, "wheel": 0, "foot": 0,
           "spring": 3, "tail": 0, "skid": 0, "spike": 0, "caster": 0}


class Actuator(NamedTuple):
    """One entry of the actuator layout, in model order."""
    slot: int       # stable node slot id
    kind: str       # "joint" or "wheel"
    axis: str       # pitch / yaw / roll for joints, "wheel" for wheels


ActuatorLayout = List[Actuator]


@dataclass
class Node:
    id:       int
    kind:     str
    parent:   int = -1
    slot:     int = 0              # slot used on the parent
    rot:      float = 0.0          # roll of the slot, radians
    axis:     str = "pitch"        # joint and spring only
    length:   float = 0.08         # joint, link and spring
    wheel_rad: float = 0.045       # wheel and caster radius
    flip:     bool = False         # joint and spring only
    width:    float = 0.020        # joint, link and spring
    thick:    float = 0.006        # joint and link; spring: blade thickness
    foot_rad: float = 0.02         # foot only
    tail_mass: float = 0.0         # tail only, counterweight mass
    offset:   float = 0.5          # attachment fraction along the parent slot
    angle:    float = 0.5          # attachment yaw / elbow angle fraction


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def section_radius(node: Node) -> float:
    """Radius of a link cross-section (width x thickness), for contact and reach."""
    return float(np.hypot(node.width, plate_thick(node)) / 2.0)


def elbow_distance(node: Node) -> float:
    """Distance from the link axis to the side slot of an elbow child."""
    return _scale(node.offset, *ELBOW_RANGE)


def elbow_angle(node: Node) -> float:
    """Angle of the side slot in the link plane, radians."""
    return _scale(node.angle, np.radians(30.0), np.radians(150.0))


SPRING_SEGMENT_THICK = 0.006   # m, the PLA link printed after a spring's TPU blade


def plate_thick(node: Node) -> float:
    """Thickness of the printed PLA plate of a limb segment. A spring's THICK
    gene is its TPU blade, so its PLA segment uses a fixed thickness."""
    return SPRING_SEGMENT_THICK if node.kind == "spring" else node.thick


def spring_stiffness(node: Node) -> float:
    """Rotational stiffness of a spring, N*m/rad: a TPU blade bending at its
    base, k = E * (width * blade^3 / 12) / free length. THICK is the blade."""
    inertia = node.width * node.thick ** 3 / 12.0
    return E_TPU * inertia / SPRING_FLEX_LEN


@dataclass
class Morphology:
    body_length: float
    body_width:  float
    nodes: Dict[int, Node] = field(default_factory=dict)

    # structure ----------------------------------------------------------
    def child_map(self) -> Dict[int, Dict[int, int]]:
        out: Dict[int, Dict[int, int]] = {}
        for n in self.nodes.values():
            if n.parent >= 0:
                out.setdefault(n.parent, {})[n.slot] = n.id
        return out

    @property
    def order(self) -> List[int]:
        """Node ids in depth first order, children by slot. Model order."""
        cm = self.child_map()
        out: List[int] = []

        def visit(i):
            out.append(i)
            for s in sorted(cm.get(i, {})):
                visit(cm[i][s])
        visit(0)
        return out

    def depth(self, i: int) -> int:
        d = 0
        while self.nodes[i].parent >= 0:
            i = self.nodes[i].parent
            d += 1
        return d

    @property
    def max_depth(self) -> int:
        return max(self.depth(i) for i in self.nodes)

    def free_slots(self) -> List[Tuple[int, int]]:
        taken = {(n.parent, n.slot) for n in self.nodes.values() if n.parent >= 0}
        return [(i, s) for i, n in self.nodes.items()
                for s in range(N_SLOTS[n.kind]) if (i, s) not in taken]

    def copy(self) -> "Morphology":
        return Morphology(self.body_length, self.body_width,
                          {i: Node(**vars(n)) for i, n in self.nodes.items()})

    # counts ------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_joints(self) -> int:
        return sum(n.kind == "joint" for n in self.nodes.values())

    @property
    def n_wheels(self) -> int:
        return sum(n.kind == "wheel" for n in self.nodes.values())

    @property
    def n_springs(self) -> int:
        return sum(n.kind == "spring" for n in self.nodes.values())

    @property
    def n_servos(self) -> int:
        return self.n_joints + self.n_wheels

    def layout(self) -> ActuatorLayout:
        out = []
        for i in self.order:
            n = self.nodes[i]
            if n.kind == "joint":
                out.append(Actuator(i, "joint", n.axis))
            elif n.kind == "wheel":
                out.append(Actuator(i, "wheel", "wheel"))
        return out

    # geometry (rest pose) ------------------------------------------------
    def slot_frame(self, parent: Node, slot: int,
                   child: Optional[Node] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Offset and rotation of a slot in the frame of its parent. The
        child's offset and angle place it along the slot; tips ignore them."""
        offset = 0.5 if child is None else child.offset
        angle = 0.5 if child is None else child.angle
        yaw = (2.0 * angle - 1.0) * np.radians(60.0)
        if parent.kind == "core":
            y = (2.0 * offset - 1.0) * 0.8 * self.body_width / 2
            if slot == 0:
                return np.array([self.body_length / 2, y, 0]), _rz(yaw)
            if slot == 1:
                return np.array([-self.body_length / 2, y, 0]), _rz(np.pi + yaw)
            side = 1.0 if slot < 5 else -1.0
            x = (SIDE_X[(slot - 2) % 3]
                 + 0.3 * (2.0 * offset - 1.0)) * self.body_length / 2
            return (np.array([x, side * self.body_width / 2, 0]),
                    _rz(side * np.pi / 2 + yaw))
        if slot == 0:
            return np.array([parent.length, 0, 0]), _rz(0.0)
        side = 1.0 if slot == 1 else -1.0
        d = elbow_distance(child) if child is not None else _scale(0.5, *ELBOW_RANGE)
        a = elbow_angle(child) if child is not None else _scale(
            0.5, np.radians(30.0), np.radians(150.0))
        return (np.array([parent.length, side * d, 0]), _rz(side * a))

    def rel_frame(self, node: Node) -> Tuple[np.ndarray, np.ndarray]:
        """Offset and rotation of a node in the frame of its parent."""
        off, rot = self.slot_frame(self.nodes[node.parent], node.slot, node)
        return off, rot @ _rx(node.rot)

    def frames(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        out = {0: (np.zeros(3), np.eye(3))}
        for i in self.order[1:]:
            n = self.nodes[i]
            pp, pr = out[n.parent]
            off, rot = self.rel_frame(n)
            out[i] = (pp + pr @ off, pr @ rot)
        return out

    def wheel_axis_sign(self, rot: np.ndarray) -> float:
        """Sign of the local Y axle so that positive speed rolls forward
        (spin about world +Y)."""
        return -1.0 if rot[1, 1] < -0.5 else 1.0

    def wheel_hub(self, node: Node, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """Wheel centre offset from the node origin, in the node frame: the
        wheel sits on the outer side of its servo."""
        half_t = max(0.01, 0.25 * node.wheel_rad)
        s = 1.0
        if abs(rot[1, 1]) > 0.5:
            s = np.sign(rot[1, 1]) * (-1.0 if pos[1] < 0 else 1.0)
        return np.array([0.0, s * (SERVO_BOX[1] / 2 + half_t), 0.0])

    def caster_hub(self, node: Node, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """Caster wheel centre offset from the node origin, in the node frame:
        the wheel hangs below its fork, on the outer side."""
        half_t = CASTER_WHEEL_T / 2
        s = 1.0
        if abs(rot[1, 1]) > 0.5:
            s = np.sign(rot[1, 1]) * (-1.0 if pos[1] < 0 else 1.0)
        return np.array([0.0, s * (CASTER_FORK_SIZE[1] / 2 + half_t),
                         -CASTER_FORK_SIZE[2] / 2 - node.wheel_rad])

    def terminal_points(self) -> List[Tuple[int, np.ndarray, float]]:
        """
        Ground contact candidates at rest, torso frame: (node id, point,
        radius) for every wheel, foot and other terminal module, and for every
        free tip of a joint, link or spring.
        """
        fr = self.frames()
        cm = self.child_map()
        out = []
        for i in self.order:
            n = self.nodes[i]
            pos, rot = fr[i]
            if n.kind == "wheel":
                out.append((i, pos + rot @ self.wheel_hub(n, pos, rot), n.wheel_rad))
            elif n.kind == "caster":
                out.append((i, pos + rot @ self.caster_hub(n, pos, rot),
                            n.wheel_rad))
            elif n.kind == "foot":
                out.append((i, pos.copy(), n.foot_rad))
            elif n.kind == "tail":
                out.append((i, pos + rot @ np.array([n.length, 0, 0]),
                            TAIL_TIP_RADIUS))
            elif n.kind == "skid":
                out.append((i, pos.copy(), SKID_RADIUS))
            elif n.kind == "spike":
                out.append((i, pos + rot @ np.array([SPIKE_LENGTH, 0, 0]),
                            SPIKE_RADIUS))
            elif n.kind in ("joint", "link", "spring") and i not in cm:
                out.append((i, pos + rot @ np.array([n.length, 0, 0]),
                            section_radius(n)))
        return out

    @property
    def n_terminals(self) -> int:
        return len(self.terminal_points())

    @property
    def max_reach(self) -> float:
        """Largest distance from the torso centre to any point of the body."""
        r = float(np.linalg.norm([self.body_length / 2, self.body_width / 2,
                                  TORSO_HEIGHT / 2]))
        for i, (pos, rot) in self.frames().items():
            n = self.nodes[i]
            r = max(r, float(np.linalg.norm(pos)))
            if n.kind in ("joint", "link", "spring"):
                tip = pos + rot @ np.array([n.length, 0, 0])
                r = max(r, float(np.linalg.norm(tip)) + section_radius(n))
        for _, p, rad in self.terminal_points():
            r = max(r, float(np.linalg.norm(p)) + rad)
        return r

    def lowest_z(self) -> float:
        """Lowest point of the body relative to the torso centre at rest."""
        low = -TORSO_HEIGHT / 2
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)]) * (np.array(SERVO_BOX) / 2)
        for i, (pos, rot) in self.frames().items():
            n = self.nodes[i]
            if n.kind in ("joint", "wheel"):
                low = min(low, float((pos + corners @ rot.T)[:, 2].min()))
            if n.kind in ("joint", "link", "spring"):
                tip = pos + rot @ np.array([n.length, 0, 0])
                r = section_radius(n)
                low = min(low, pos[2] - r, tip[2] - r)
            elif n.kind == "wheel":
                c = pos + rot @ self.wheel_hub(n, pos, rot)
                az = abs(float((rot @ np.array([0.0, 1.0, 0.0]))[2]))
                half = max(0.01, 0.25 * n.wheel_rad)
                low = min(low, float(c[2] - half * az
                                     - n.wheel_rad * np.sqrt(max(0.0, 1 - az * az))))
            elif n.kind == "caster":
                c = pos + rot @ self.caster_hub(n, pos, rot)
                az = abs(float((rot @ np.array([0.0, 1.0, 0.0]))[2]))
                half = CASTER_WHEEL_T / 2
                low = min(low, float(pos[2] - CASTER_FORK_SIZE[2] / 2),
                          float(c[2] - half * az
                                - n.wheel_rad * np.sqrt(max(0.0, 1 - az * az))))
            elif n.kind == "foot":
                low = min(low, pos[2] - n.foot_rad)
            elif n.kind == "tail":
                tip = pos + rot @ np.array([n.length, 0, 0])
                low = min(low, pos[2] - TAIL_BAR_SIZE[1] / 2,
                          tip[2] - TAIL_TIP_RADIUS)
            elif n.kind == "skid":
                low = min(low, pos[2] - SKID_SIZE[2] / 2)
            elif n.kind == "spike":
                tip = pos + rot @ np.array([SPIKE_LENGTH, 0, 0])
                low = min(low, pos[2] - SPIKE_RADIUS, tip[2] - SPIKE_RADIUS)
        return low


# --------------------------------------------------------------------------
# Genome <-> tree
# --------------------------------------------------------------------------

def _pick(u: float, n: int) -> int:
    return min(int(np.clip(u, 0.0, 1.0) * n), n - 1)


def _bin(i: int, n: int) -> float:
    return (i + 0.5) / n


def _scale(u: float, lo: float, hi: float) -> float:
    return lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo)


def _unscale(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def violations(morph: Morphology) -> List[str]:
    """Printability checks: link bending strength, bed footprint, elbow reach.

    Torso size is not checked here: it will be split into tiles.
    """
    out: List[str] = []
    for i, n in morph.nodes.items():
        if n.kind not in ("joint", "link", "spring"):
            continue
        modulus = n.width * plate_thick(n) ** 2 / 6.0
        stress = SERVO_STALL / modulus if modulus > 0 else np.inf
        limit = PLA_STRENGTH * STRENGTH_SAFETY
        if stress > limit:
            out.append(f"#{i} {n.kind}: bending stress {stress / 1e6:.1f} MPa "
                       f"above the {limit / 1e6:.1f} MPa limit")
        if n.length + n.width > BED + 1e-9:
            out.append(f"#{i} {n.kind}: {n.length * 1000:.0f} + "
                       f"{n.width * 1000:.0f} mm does not fit the "
                       f"{BED * 1000:.0f} mm bed")
        if n.slot in (1, 2) and morph.nodes[n.parent].kind in (
                "joint", "link", "spring"):
            parent_width = morph.nodes[n.parent].width
            if elbow_distance(n) + parent_width > BED + 1e-9:
                out.append(f"#{i} {n.kind}: elbow {elbow_distance(n) * 1000:.0f} + "
                           f"{parent_width * 1000:.0f} mm does not fit the "
                           f"{BED * 1000:.0f} mm bed")
    return out


def decode(genes: np.ndarray) -> Morphology:
    """Genome -> body. Total: any vector in [0, 1] gives a valid body."""
    genes = np.asarray(genes, dtype=float)
    assert genes.shape == (MORPH_LENGTH,), genes.shape
    blocks = genes.reshape(MAX_NODES, BLOCK_GENES)

    morph = Morphology(round(_scale(blocks[0, LENGTH], *BODY_BOUNDS[0]), 6),
                       round(_scale(blocks[0, WHEEL], *BODY_BOUNDS[1]), 6),
                       {0: Node(0, "core")})
    kinds = {0: "core"}
    for i in range(1, MAX_NODES):
        kinds[i] = TYPE_KINDS[_pick(blocks[i, TYPE], len(TYPE_KINDS))]

    # Who claims which slot: the lowest block id wins a clash.
    claims: Dict[Tuple[int, int], int] = {}
    for i in range(1, MAX_NODES):
        b = blocks[i]
        if b[ACTIVE] <= 0.5:
            continue
        p = _pick(b[PARENT], MAX_NODES)
        if p == i or N_SLOTS[kinds[p]] == 0:
            continue
        claims.setdefault((p, _pick(b[SLOT], N_SLOTS[kinds[p]])), i)
    kids: Dict[int, List[Tuple[int, int]]] = {}
    for (p, s), i in claims.items():
        kids.setdefault(p, []).append((s, i))

    servos = 0
    frontier = [0]
    for _ in range(MAX_DEPTH):
        nxt = []
        for p in frontier:
            for s, i in sorted(kids.get(p, [])):
                kind = kinds[i]
                if kind in ("joint", "wheel"):
                    if servos >= MAX_SERVOS:
                        continue
                    servos += 1
                morph.nodes[i] = _node_from_block(i, kind, p, s, blocks[i])
                nxt.append(i)
        frontier = nxt
    _ensure_servo(morph)
    return morph


def _node_from_block(i, kind, parent, slot, b) -> Node:
    thick_range = SPRING_BLADE_RANGE if kind == "spring" else LINK_THICK_RANGE
    n = Node(i, kind, parent, slot,
             rot=float((np.clip(b[ROT], 0.0, 1.0) % 1.0) * 2 * np.pi),
             width=round(_scale(b[WIDTH], *LINK_WIDTH_RANGE), 6),
             thick=round(_scale(b[THICK], *thick_range), 6),
             offset=round(_scale(b[OFFSET], 0.0, 1.0), 6),
             angle=round(_scale(b[ANGLE], 0.0, 1.0), 6))
    if kind in ("joint", "spring"):
        n.axis = AXES[_pick(b[AXIS], 3)]
        n.flip = bool(b[FLIP] > 0.5)
    if kind in ("joint", "link", "spring", "tail"):
        n.length = round(_scale(b[LENGTH], *LENGTH_RANGE), 6)
    if kind == "wheel":
        n.wheel_rad = round(_scale(b[WHEEL], *WHEEL_RANGE_M), 6)
    if kind == "caster":
        n.wheel_rad = round(_scale(b[WHEEL], *CASTER_RANGE), 6)
    if kind == "foot":
        n.foot_rad = round(_scale(b[WHEEL], *FOOT_RANGE), 6)
    if kind == "tail":
        n.tail_mass = round(_scale(b[WHEEL], *TAIL_MASS_RANGE), 6)
    return n


def encode(morph: Morphology) -> np.ndarray:
    """Body -> canonical genome (inverse of decode)."""
    blocks = np.zeros((MAX_NODES, BLOCK_GENES))
    blocks[0, ACTIVE] = 1.0
    blocks[0, LENGTH] = _unscale(morph.body_length, *BODY_BOUNDS[0])
    blocks[0, WHEEL] = _unscale(morph.body_width, *BODY_BOUNDS[1])
    for i, n in morph.nodes.items():
        if i == 0:
            continue
        b = blocks[i]
        b[ACTIVE] = 1.0
        b[PARENT] = _bin(n.parent, MAX_NODES)
        b[SLOT] = _bin(n.slot, N_SLOTS[morph.nodes[n.parent].kind])
        b[ROT] = _unscale(n.rot, 0.0, 2 * np.pi)
        b[TYPE] = _bin(TYPE_KINDS.index(n.kind), len(TYPE_KINDS))
        b[AXIS] = _bin(AXES.index(n.axis), 3)
        b[LENGTH] = _unscale(n.length, *LENGTH_RANGE)
        if n.kind == "foot":
            b[WHEEL] = _unscale(n.foot_rad, *FOOT_RANGE)
        elif n.kind == "tail":
            b[WHEEL] = _unscale(n.tail_mass, *TAIL_MASS_RANGE)
        elif n.kind == "caster":
            b[WHEEL] = _unscale(n.wheel_rad, *CASTER_RANGE)
        else:
            b[WHEEL] = _unscale(n.wheel_rad, *WHEEL_RANGE_M)
        b[FLIP] = 0.75 if n.flip else 0.25
        b[WIDTH] = _unscale(n.width, *LINK_WIDTH_RANGE)
        b[THICK] = _unscale(n.thick, *(SPRING_BLADE_RANGE if n.kind == "spring"
                                       else LINK_THICK_RANGE))
        b[OFFSET] = _unscale(n.offset, 0.0, 1.0)
        b[ANGLE] = _unscale(n.angle, 0.0, 1.0)
    return blocks.reshape(-1)


def repair(genes: np.ndarray) -> np.ndarray:
    """Canonical genome of the body the genes decode to."""
    return encode(decode(genes))


def _free_id(morph: Morphology) -> Optional[int]:
    for i in range(1, MAX_NODES):
        if i not in morph.nodes:
            return i
    return None


def _ensure_servo(morph: Morphology):
    """A body without servos is a brick: hang a default joint on the core."""
    if morph.n_servos:
        return
    i = _free_id(morph)
    taken = morph.child_map().get(0, {})
    slot = next(s for s in (3, 2, 4, 6, 5, 7, 0, 1) if s not in taken)
    morph.nodes[i] = Node(i, "joint", 0, slot, 3 * np.pi / 2)


def prune(morph: Morphology) -> Morphology:
    """Enforce the limits in place by dropping deepest leaves first."""
    while True:
        parents = {n.parent for n in morph.nodes.values()}
        leaves = [i for i in morph.nodes if i != 0 and i not in parents]
        if morph.max_depth > MAX_DEPTH:
            drop = [i for i in leaves if morph.depth(i) > MAX_DEPTH]
        elif morph.n_nodes > MAX_NODES:
            drop = leaves
        elif morph.n_servos > MAX_SERVOS:
            drop = [i for i in leaves if morph.nodes[i].kind in ("joint", "wheel")]
            drop = drop or leaves
        else:
            break
        del morph.nodes[max(drop, key=lambda i: (morph.depth(i), i))]
    _ensure_servo(morph)
    return morph


# --------------------------------------------------------------------------
# Mirroring and templates
# --------------------------------------------------------------------------

def mirror_slot(kind: str, slot: int) -> int:
    """Slot on the other side of the XZ plane."""
    if kind == "core":
        return slot + 3 if 2 <= slot <= 4 else slot - 3 if slot >= 5 else slot
    return {1: 2, 2: 1}.get(slot, slot)


def copy_subtree(morph: Morphology, root: int, parent: int, slot: int,
                 mirror: bool = False) -> Optional[int]:
    """
    Copy the subtree under `root` onto a free slot of `parent`, breadth first,
    stopping where a limit is hit (so a partial copy is still a valid body).
    mirror=True reflects it across the XZ plane: sides swapped, rolls
    negated, pitch and roll axes flipped. Returns the new root id or None.
    """
    if (parent, slot) in {(n.parent, n.slot) for n in morph.nodes.values()}:
        return None
    base = morph.depth(parent) + 1 - morph.depth(root)
    cm = morph.child_map()
    queue = [(root, parent, slot)]
    new_root = None
    while queue:
        src, par, sl = queue.pop(0)
        node = morph.nodes[src]
        i = _free_id(morph)
        if i is None or base + morph.depth(src) > MAX_DEPTH:
            continue
        if node.kind in ("joint", "wheel") and morph.n_servos >= MAX_SERVOS:
            continue
        new = Node(**vars(node))
        new.id, new.parent, new.slot = i, par, sl
        if mirror:
            new.rot = (-node.rot) % (2 * np.pi)
            if par == 0:
                new.angle = 1.0 - node.angle      # yaw mirrors around the torso
            if node.kind in ("joint", "spring") and node.axis != "yaw":
                new.flip = not node.flip
        morph.nodes[i] = new
        new_root = i if new_root is None else new_root
        for s, c in sorted(cm.get(src, {}).items()):
            queue.append((c, i, mirror_slot(node.kind, s) if mirror else s))
    return new_root


def template(n_pairs: int = 2, wheels: bool = False, yaw: bool = False,
             thigh: float = 0.10, shin: float = 0.10, wheel_rad: float = 0.05,
             body_length: float = 0.40, body_width: float = 0.20) -> Morphology:
    """
    Mirrored leg pairs on the core sides, like the old fixed genome:
    [yaw joint] -> hip pitch (thigh) -> elbow -> knee pitch (shin) -> [wheel].
    The right leg is a mirror copy of the left one; pairs beyond MAX_NODES or
    MAX_SERVOS are left out. Positive pitch lifts the
    thigh and swings the shin outward, positive yaw swings the foot forward.
    """
    morph = Morphology(body_length, body_width, {0: Node(0, "core")})
    chain = []                       # (kind, slot on the parent, fields)
    if yaw:
        chain.append(("joint", 0, dict(axis="yaw", length=0.06)))
    chain.append(("joint", 0, dict(axis="pitch", length=thigh, flip=True)))
    chain.append(("joint", 1, dict(axis="pitch", length=shin, flip=True,
                                   offset=ELBOW_OFFSET)))
    if wheels:
        chain.append(("wheel", 0, dict(wheel_rad=wheel_rad)))
    for core_slot in {1: (3,), 2: (2, 4), 3: (2, 3, 4)}[n_pairs]:
        servos = sum(k in ("joint", "wheel") for k, _, _ in chain)
        if (morph.n_nodes + 2 * len(chain) > MAX_NODES
                or morph.n_servos + 2 * servos > MAX_SERVOS):
            break                    # a pair that does not fit is left out
        parent, first = 0, None
        for kind, slot, fields in chain:
            i = _free_id(morph)
            first = i if first is None else first
            morph.nodes[i] = Node(i, kind, parent, core_slot if parent == 0 else slot,
                                  3 * np.pi / 2 if parent == 0 else 0.0, **fields)
            parent = i
        copy_subtree(morph, first, 0, mirror_slot("core", core_slot), mirror=True)
    return morph


def _random_seed(rng: np.random.Generator) -> Morphology:
    wheels = bool(rng.random() < 0.35)
    yaw = bool(rng.random() < 0.5)
    n_pairs = int(rng.integers(1, 4))
    while n_pairs > 1 and 2 * n_pairs * (2 + yaw + wheels) > MAX_SERVOS:
        n_pairs -= 1
    return template(n_pairs, wheels, yaw,
                    thigh=round(float(rng.uniform(0.06, 0.12)), 6),
                    shin=round(float(rng.uniform(0.06, 0.12)), 6),
                    wheel_rad=round(float(rng.uniform(0.03, 0.05)), 6),
                    body_length=round(float(rng.uniform(
                        0.30 if n_pairs > 1 else 0.15, 0.40)), 6),
                    body_width=round(float(rng.uniform(*BODY_BOUNDS[1])), 6))


# --------------------------------------------------------------------------
# Random bodies and mutation
# --------------------------------------------------------------------------

def _random_node(rng, kind: str, parent: int, slot: int) -> Node:
    n = Node(0, kind, parent, slot, float(rng.uniform(0, 2 * np.pi)))
    if kind in ("joint", "spring"):
        n.axis = str(rng.choice(AXES, p=[0.6, 0.25, 0.15]))
        n.flip = bool(rng.random() < 0.5)
    if kind in ("joint", "link", "spring", "tail"):
        n.length = round(float(rng.uniform(0.06, 0.12)), 6)
    if kind == "spring":
        n.thick = round(float(rng.uniform(*SPRING_BLADE_RANGE)), 6)
    if kind == "tail":
        n.tail_mass = round(float(rng.uniform(*TAIL_MASS_RANGE)), 6)
    if kind == "wheel":
        n.wheel_rad = round(float(rng.uniform(*WHEEL_RANGE_M)), 6)
    if kind == "caster":
        n.wheel_rad = round(float(rng.uniform(*CASTER_RANGE)), 6)
    if kind == "foot":
        n.foot_rad = round(float(rng.uniform(*FOOT_RANGE)), 6)
    return n


def _add_random(morph: Morphology, rng, p: int, s: int) -> Optional[int]:
    i = _free_id(morph)
    if i is None or morph.depth(p) + 1 > MAX_DEPTH:
        return None
    kind = str(rng.choice(TYPE_KINDS,
                          p=[0.45, 0.14, 0.10, 0.18, 0.05, 0.03, 0.02,
                             0.02, 0.01]))
    if kind in ("joint", "wheel") and morph.n_servos >= MAX_SERVOS:
        kind = "link" if rng.random() < 0.5 else "foot"
    node = _random_node(rng, kind, p, s)
    node.id = i
    morph.nodes[i] = node
    return i


def _grow_limb(morph: Morphology, rng, slot: int) -> Optional[int]:
    """A chain of 1 to 3 joints or links from a core slot, maybe a wheel/foot."""
    parent, sl, first = 0, slot, None
    for step in range(int(rng.integers(1, 4))):
        i = _free_id(morph)
        if i is None or morph.n_servos >= MAX_SERVOS:
            break
        node = _random_node(rng, "joint" if rng.random() < 0.8 else "link", parent, sl)
        if step == 0 and rng.random() < 0.7:
            node.rot = 3 * np.pi / 2   # side limbs start pointing down
        node.id = i
        morph.nodes[i] = node
        first = i if first is None else first
        parent, sl = i, (0 if rng.random() < 0.4 else int(rng.integers(1, 3)))
    if parent and rng.random() < 0.5:
        _add_random(morph, rng, parent, sl)
    return first


def _random_tree(rng: np.random.Generator) -> Morphology:
    morph = Morphology(round(float(rng.uniform(*BODY_BOUNDS[0])), 6),
                       round(float(rng.uniform(*BODY_BOUNDS[1])), 6),
                       {0: Node(0, "core")})
    for _ in range(int(rng.integers(1, 4))):
        slot = int(rng.choice([2, 3, 4, 0, 1], p=[.25, .15, .25, .2, .15]))
        if slot in morph.child_map().get(0, {}):
            continue
        first = _grow_limb(morph, rng, slot)
        if first is not None and slot >= 2 and rng.random() < 0.8:
            copy_subtree(morph, first, 0, mirror_slot("core", slot), mirror=True)
    return prune(morph)


def random_morphology(rng: np.random.Generator) -> np.ndarray:
    """Half template seeds (mirrored leg pairs), half small random trees."""
    morph = _random_seed(rng) if rng.random() < 0.5 else _random_tree(rng)
    return encode(prune(morph))


def _jitter(value: float, lo: float, hi: float, rng) -> float:
    """Gaussian step in normalised units, clipped to the gene range."""
    u = _unscale(value, lo, hi) + rng.normal(0.0, 0.1)
    return round(_scale(float(np.clip(u, 0.0, 1.0)), lo, hi), 6)


def _op_param(morph, rng):
    n = morph.nodes[int(rng.choice(list(morph.nodes)))]
    if n.kind == "core":
        if rng.random() < 0.5:
            morph.body_length = _jitter(morph.body_length, *BODY_BOUNDS[0], rng)
        else:
            morph.body_width = _jitter(morph.body_width, *BODY_BOUNDS[1], rng)
        return
    choices = ["rot", "offset", "angle"]
    if n.kind in ("joint", "link", "spring"):
        choices += ["length"] * 2 + ["width", "thick"]
    if n.kind == "tail":
        choices += ["length"] * 2 + ["tail_mass"]
    if n.kind in ("joint", "spring"):
        choices += ["axis", "flip"]
    if n.kind in ("wheel", "caster"):
        choices += ["wheel_rad"] * 2
    if n.kind == "foot":
        choices += ["foot_rad"]
    what = str(rng.choice(choices))
    if what == "rot":
        n.rot = _jitter(n.rot, 0.0, 2 * np.pi, rng) % (2 * np.pi)
    elif what == "length":
        n.length = _jitter(n.length, *LENGTH_RANGE, rng)
    elif what == "width":
        n.width = _jitter(n.width, *LINK_WIDTH_RANGE, rng)
    elif what == "thick":
        n.thick = _jitter(n.thick, *(SPRING_BLADE_RANGE if n.kind == "spring"
                                     else LINK_THICK_RANGE), rng)
    elif what == "offset":
        n.offset = _jitter(n.offset, 0.0, 1.0, rng)
    elif what == "angle":
        n.angle = _jitter(n.angle, 0.0, 1.0, rng)
    elif what == "axis":
        n.axis = str(rng.choice(AXES))
    elif what == "flip":
        n.flip = not n.flip
    elif what == "wheel_rad":
        span = CASTER_RANGE if n.kind == "caster" else WHEEL_RANGE_M
        n.wheel_rad = _jitter(n.wheel_rad, *span, rng)
    elif what == "tail_mass":
        n.tail_mass = _jitter(n.tail_mass, *TAIL_MASS_RANGE, rng)
    else:
        n.foot_rad = _jitter(n.foot_rad, *FOOT_RANGE, rng)


def _op_type(morph, rng):
    """Swap the type of a limb segment, or of a leaf (also wheel and foot)."""
    parents = {n.parent for n in morph.nodes.values()}
    cand = [i for i, n in morph.nodes.items() if i != 0
            and (n.kind in ("joint", "link", "spring") or i not in parents)]
    if not cand:
        return
    i = int(rng.choice(cand))
    n = morph.nodes[i]
    kinds = ["joint", "link", "spring"] + (
        ["wheel", "foot", "tail", "skid", "spike", "caster"]
        if i not in parents else [])
    kind = str(rng.choice([k for k in kinds if k != n.kind]))
    if kind in ("joint", "wheel") and n.kind not in ("joint", "wheel") \
            and morph.n_servos >= MAX_SERVOS:
        return
    new = _random_node(rng, kind, n.parent, n.slot)
    new.id, new.rot = i, n.rot
    new.offset, new.angle = n.offset, n.angle
    if kind in ("joint", "link", "spring", "tail") \
            and n.kind in ("joint", "link", "spring", "tail"):
        new.length = n.length
    if kind in ("joint", "link", "spring") \
            and n.kind in ("joint", "link", "spring"):
        new.width = n.width
        range_ = SPRING_BLADE_RANGE if kind == "spring" else LINK_THICK_RANGE
        new.thick = round(float(np.clip(n.thick, *range_)), 6)
    morph.nodes[i] = new


def _op_add(morph, rng):
    """A new module, sometimes with one more behind it, in a free slot."""
    free = [fs for fs in morph.free_slots() if morph.depth(fs[0]) < MAX_DEPTH]
    if not free:
        return
    p, s = free[int(rng.integers(len(free)))]
    i = _add_random(morph, rng, p, s)
    if i is not None and morph.nodes[i].kind in ("joint", "link", "spring") \
            and rng.random() < 0.4:
        slots = [(p2, s2) for p2, s2 in morph.free_slots() if p2 == i]
        if slots:
            _add_random(morph, rng, *slots[int(rng.integers(len(slots)))])


def _op_remove(morph, rng):
    cand = [i for i in morph.nodes if i != 0]
    if not cand:
        return
    cm = morph.child_map()
    stack, gone = [int(rng.choice(cand))], []
    while stack:
        i = stack.pop()
        gone.append(i)
        stack.extend(cm.get(i, {}).values())
    for i in gone:
        del morph.nodes[i]


def _op_mirror(morph, rng):
    """Duplicate a subtree on the opposite side (left/right)."""
    taken = {(n.parent, n.slot) for n in morph.nodes.values()}
    cand = []
    for i, n in morph.nodes.items():
        if i == 0:
            continue
        ms = mirror_slot(morph.nodes[n.parent].kind, n.slot)
        if ms != n.slot and (n.parent, ms) not in taken:
            cand.append((i, ms))
    if cand:
        i, ms = cand[int(rng.integers(len(cand)))]
        copy_subtree(morph, i, morph.nodes[i].parent, ms, mirror=True)


def _op_copy(morph, rng):
    """Copy a subtree to another free slot."""
    cand = [i for i in morph.nodes if i != 0]
    free = morph.free_slots()
    if not cand or not free:
        return
    i = int(rng.choice(cand))
    p, s = free[int(rng.integers(len(free)))]
    q = p
    while q >= 0 and q != i:          # a subtree cannot grow inside itself
        q = morph.nodes[q].parent
    if q != i:
        copy_subtree(morph, i, p, s)


MUTATIONS = ((_op_param, 0.35), (_op_type, 0.15), (_op_add, 0.2),
             (_op_remove, 0.1), (_op_mirror, 0.15), (_op_copy, 0.05))


def mutate(genes: np.ndarray, rng: np.random.Generator,
           n_ops: Optional[int] = None) -> np.ndarray:
    """Apply 1 to 3 tree operators to the decoded body, then repair."""
    morph = decode(genes)
    probs = np.array([p for _, p in MUTATIONS])
    for _ in range(n_ops or int(rng.integers(1, 4))):
        op, _ = MUTATIONS[int(rng.choice(len(MUTATIONS), p=probs / probs.sum()))]
        op(morph, rng)
        prune(morph)
    return encode(morph)


# --------------------------------------------------------------------------
# Descriptors and summary
# --------------------------------------------------------------------------

def descriptor(m: Morphology) -> tuple:
    """Features used as MAP-Elites axes: (limb pairs, wheeled pairs, servos)."""
    # TODO D3: the map moves to behaviour descriptors.
    pairs = int(np.clip(round(m.n_terminals / 2), 1, 3))
    return (pairs, int(np.clip(m.n_wheels // 2, 0, 3)),
            int(np.clip(m.n_servos, 4, 24)))


def summary(m: Morphology) -> str:
    lines = [f"torso {m.body_length:.2f} x {m.body_width:.2f} m, "
             f"{m.n_nodes} modules, {m.n_servos} servos, {m.n_wheels} wheels, "
             f"{m.n_terminals} ground points, depth {m.max_depth}"]
    cm = m.child_map()

    def show(i, indent):
        n = m.nodes[i]
        desc = "core"
        if n.kind != "core":
            desc = n.kind
            if n.kind in ("joint", "spring"):
                desc += f" {n.axis}{'-' if n.flip else ''} {n.length * 1000:.0f}mm"
            elif n.kind == "link":
                desc += f" {n.length * 1000:.0f}mm"
            elif n.kind in ("wheel", "caster"):
                desc += f" r={n.wheel_rad * 1000:.0f}mm"
            elif n.kind == "tail":
                desc += f" {n.length * 1000:.0f}mm {n.tail_mass * 1000:.0f}g"
            desc += f" (slot {n.slot}, roll {np.degrees(n.rot):.0f} deg)"
        lines.append("  " * indent + f"#{i} {desc}")
        for s in sorted(cm.get(i, {})):
            show(cm[i][s], indent + 1)
    show(0, 1)
    return "\n".join(lines)

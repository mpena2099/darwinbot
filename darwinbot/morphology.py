"""
morphology.py
-------------
Body genome: a tree of real modules (Karl Sims / Revolve2 style).

Modules
  core   root, torso plate + ESP32 + LiPo. The torso is an 8-vertex star-shaped
         polygon extruded to TORSO_HEIGHT, one vertex per old box direction
         (front, corners and side midpoints). 8 slots sit on the contour:
         0 front, 1 back, 2..4 left side (front, middle, rear), 5..7 right side.
         4 more sit on the flat faces: 8 top-front, 9 top-rear, 10 bottom-front,
         11 bottom-rear. A face child points straight up or down and ANGLE
         yaws it about the vertical by +/-180 degrees; OFFSET slides it across
         the torso (a bottom mount may not sit over the electronics)
  segment a spine servo (a hinge like a joint) at the node origin followed by a
         second torso plate, a LENGTH x WIDTH x SEGMENT_HEIGHT box. Its 8 slots
         use the core placement rule on that plate; slot 1 faces the parent and
         is never used. Segments attach only to core or segment slots 0 and 1,
         at most MAX_SEGMENTS per body, and carry no electronics
  joint  one STS3215 servo + a printed link. Slots: 0 tip, 1 left, 2 right
         (the side slots are an elbow at the end of the link)
  link   passive printed link, same slots as the joint
  spring passive TPU flexure: a passive hinge plus a printed link, same slots
         as a link. It bends only about its blade's thin axis (local Y), so the
         AXIS gene is ignored and the slot roll sets the bending direction
  wheel  servo in wheel mode + wheel, terminal
  foot   TPU tip, terminal, one of FOOT_SHAPES. A joint or link with no child
         ends in an implicit spherical foot.
  tail   printed bar with a counterweight at the tip, terminal
  skid   low-friction pad, terminal
  spike  high-friction TPU tip, terminal
  caster free wheel on a passive hinge, terminal

Every slot also has a continuous roll in [0, 2*pi) around the slot direction,
so a child can point anywhere. A joint turns about its pitch (local Z), yaw
(local Y) or roll (local X) axis; local X is the direction of the link. `flip`
negates the axis, which is how a mirrored copy keeps "same signal, mirrored
motion".

A joint, link or spring keeps its path length `length` whatever its PROFILE:
  straight  one piece along local X
  L         two equal pieces, the second turned by `bend` about local Z
  arc       three equal pieces, turned by bend/2 at each inner joint
The tip is the end of that path, with the accumulated rotation (link_tip).

Genome: MAX_NODES blocks of BLOCK_GENES genes in [0, 1], one per node slot.
The block index is the stable slot id of the node (the controller is indexed
by it). Block 0 is always the core. A block is
  [ACTIVE, PARENT, SLOT, ROT, TYPE, AXIS, LENGTH, WHEEL, FLIP,
   WIDTH, THICK, OFFSET, ANGLE, PROFILE, BEND]
(for the core, LENGTH and WHEEL hold body length and width, and the unused
genes PARENT, SLOT, ROT, TYPE, AXIS, FLIP, WIDTH and THICK hold the contour
factors in CONTOUR_RANGE; PROFILE and BEND are unused). LENGTH is the
link length (40..200 mm); WIDTH and THICK are the link section (12..30 mm by
3..10 mm, and for a spring THICK is the TPU blade, 2..8 mm); WHEEL is the wheel
radius or, for a foot, the foot radius; ROT is
the slot roll in [0, 2*pi). PROFILE is one of PROFILES and BEND the signed bend
of an L or arc link in BEND_RANGE (whole degrees in the genome, radians in
code); both are ignored by other kinds. For a foot, the otherwise unused AXIS
gene picks the foot shape. OFFSET and ANGLE are attachment fractions in
[0, 1] whose meaning depends on the parent:
  * core contour slots: OFFSET slides the mount along the contour by +/-30%
    of the edge to the next or previous vertex and ANGLE yaws it by up to
    +/-60 degrees; face slots: see core above;
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
MAX_LEGS   = 8          # descriptor scale of n_legs (normalise clips)
MAX_DEPTH  = 5      # core is depth 0
MAX_SEGMENTS = 2    # spine plates per body

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
CONTOUR_RANGE = (0.4, 1.4)                # contour factor of a torso vertex
ELECTRONICS_BOX = (0.10, 0.06)            # m, ESP32 + LiPo footprint, centred
SERVO_BOX = (0.045, 0.025, 0.035)   # m, length along the link, width, height
SEGMENT_LENGTH_RANGE = (0.08, 0.22)      # m, spine plate length
SEGMENT_WIDTH_RANGE = (0.08, 0.22)       # m, spine plate width
SEGMENT_HEIGHT = 0.03                    # m, spine plate thickness
ELBOW_OFFSET = (0.04 - ELBOW_RANGE[0]) / (ELBOW_RANGE[1] - ELBOW_RANGE[0])

# Printability: a part must print on a common bed and survive the servo stall torque.
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
              "tail", "skid", "spike", "caster", "segment")    # from the TYPE gene
SERVO_JOINT_KINDS = ("joint", "segment")   # modules driven like a joint
AXES = ("pitch", "yaw", "roll")
AXIS_LOCAL = {"pitch": (0.0, 0.0, 1.0), "yaw": (0.0, 1.0, 0.0),
              "roll": (1.0, 0.0, 0.0)}
PROFILES = ("straight", "L", "arc")                     # from the PROFILE gene
BEND_RANGE = (-90.0, 90.0)      # degrees, signed bend of an L or arc link
FOOT_SHAPES = ("sphere", "blade", "pad", "claw")        # from the foot AXIS gene
CLAW_SPREAD = np.radians(30.0)  # claw toes spread around local X, in local XY
CLAW_LENGTH = 1.5               # times foot radius, length of a claw capsule
CLAW_RADIUS = 0.2               # times foot radius, radius of a claw capsule

(ACTIVE, PARENT, SLOT, ROT, TYPE, AXIS, LENGTH, WHEEL, FLIP,
 WIDTH, THICK, OFFSET, ANGLE, PROFILE, BEND) = range(15)
BLOCK_GENES  = 15
MORPH_LENGTH = MAX_NODES * BLOCK_GENES
# Core block genes that no core field needs hold the 8 contour factors.
CONTOUR_GENES = (PARENT, SLOT, ROT, TYPE, AXIS, FLIP, WIDTH, THICK)

# Core slot -> contour vertex. Slot 0 is front (+X), then counter-clockwise.
CORE_VERTEX = {0: 0, 2: 1, 3: 2, 4: 3, 1: 4, 7: 5, 6: 6, 5: 7}

# Core face slots (top-front, top-rear, bottom-front, bottom-rear) follow the
# 8 contour slots. FACE_X is the fraction of the front or rear vertex radius.
FACE_SLOTS = {8: (1.0, 1.0), 9: (-1.0, 1.0), 10: (1.0, -1.0), 11: (-1.0, -1.0)}
FACE_X = 0.5
FACE_Y = 0.8            # fraction of the half width at the mount's x

N_SLOTS = {"core": 12, "joint": 3, "link": 3, "wheel": 0, "foot": 0,
           "spring": 3, "tail": 0, "skid": 0, "spike": 0, "caster": 0,
           "segment": 8}


def is_joint(kind: str) -> bool:
    """True for every module driven as a position servo joint."""
    return kind in SERVO_JOINT_KINDS


def segment_center(node: "Node") -> np.ndarray:
    """Centre of a segment plate in the node frame."""
    return np.array([SERVO_BOX[0] / 2 + node.length / 2, 0.0, 0.0])


def segment_box(node: "Node") -> Tuple[np.ndarray, np.ndarray]:
    """Centre and half extents of a segment plate in the node frame."""
    return segment_center(node), np.array(
        [node.length / 2, node.width / 2, SEGMENT_HEIGHT / 2])


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
    foot_shape: str = "sphere"     # foot only, one of FOOT_SHAPES
    tail_mass: float = 0.0         # tail only, counterweight mass
    offset:   float = 0.5          # attachment fraction along the parent slot
    angle:    float = 0.5          # attachment yaw / elbow angle fraction
    profile:  str = "straight"     # joint, link and spring, one of PROFILES
    bend:     float = 0.0          # joint, link and spring: signed bend, radians


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _half_width(poly: List[Tuple[float, float]], x: float) -> float:
    """Half width of the torso contour at `x`: the smaller |y| of the polygon
    edge crossings of that x, so a star-shaped contour never puts a face mount
    past its own edge. 0 when the line misses the polygon."""
    ys = []
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (x1 - x) * (x2 - x) <= 0 and x1 != x2:
            ys.append(abs(y1 + (y2 - y1) * (x - x1) / (x2 - x1)))
    return min(ys) if ys else 0.0


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


def link_path(node: Node) -> List[np.ndarray]:
    """Polyline of a link in the node frame, starting at the origin. The path
    length is always `length`: one piece for a straight link, two for L and
    three for an arc, each one turned by bend / inner joints about local Z."""
    pieces = {"straight": 1, "L": 2, "arc": 3}.get(node.profile, 1)
    step = node.length / pieces
    turn = node.bend / (pieces - 1) if pieces > 1 else 0.0
    points = [np.zeros(3)]
    angle = 0.0
    for k in range(pieces):
        if k:
            angle += turn
        points.append(points[-1] + _rz(angle) @ np.array([step, 0, 0]))
    return points


def link_tip(node: Node) -> Tuple[np.ndarray, np.ndarray]:
    """End point of a link in the node frame and its accumulated rotation."""
    turn = node.bend if node.profile in ("L", "arc") else 0.0
    return link_path(node)[-1], _rz(turn)


def foot_reach(node: Node) -> float:
    """Bounding radius of a foot shape around its origin, for contact and reach.
    A pad's cylinder axis is along local X (the limb direction), so its flat
    face looks down when the limb points down."""
    r = node.foot_rad
    if node.foot_shape == "blade":
        return float(np.hypot(r, r / 2))      # 2r x r/3 x r box
    if node.foot_shape == "pad":
        return float(np.hypot(r, r / 6))      # disc radius r, height r/3
    if node.foot_shape == "claw":
        return (CLAW_LENGTH + CLAW_RADIUS) * r
    return r


def foot_lowest(node: Node, rot: np.ndarray) -> float:
    """Lowest point of a foot shape along world Z, relative to its origin."""
    r = node.foot_rad
    down = rot[2]                              # world Z in the node frame
    if node.foot_shape == "blade":
        return -float(np.abs(down) @ np.array([r, r / 6, r / 2]))
    if node.foot_shape == "pad":               # disc: axis X, half height r/6
        az = abs(float(down[0]))
        return -(r / 6 * az + r * np.sqrt(max(0.0, 1 - az * az)))
    if node.foot_shape == "claw":
        low = 0.0
        for a in (-CLAW_SPREAD, 0.0, CLAW_SPREAD):
            end = np.array([np.cos(a), np.sin(a), 0.0]) * CLAW_LENGTH * r
            low = min(low, float(down @ end))
        return low - CLAW_RADIUS * r
    return -r


@dataclass
class Morphology:
    body_length: float
    body_width:  float
    nodes: Dict[int, Node] = field(default_factory=dict)
    contour: Tuple[float, ...] = (1.0,) * 8

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
                for s in range(N_SLOTS[n.kind])
                if (i, s) not in taken and not (n.kind == "segment" and s == 1)]

    def copy(self) -> "Morphology":
        return Morphology(self.body_length, self.body_width,
                          {i: Node(**vars(n)) for i, n in self.nodes.items()},
                          tuple(self.contour))

    # counts ------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_joints(self) -> int:
        return sum(is_joint(n.kind) for n in self.nodes.values())

    @property
    def n_wheels(self) -> int:
        return sum(n.kind == "wheel" for n in self.nodes.values())

    @property
    def n_segments(self) -> int:
        return sum(n.kind == "segment" for n in self.nodes.values())

    @property
    def n_springs(self) -> int:
        return sum(n.kind == "spring" for n in self.nodes.values())

    @property
    def n_servos(self) -> int:
        return self.n_joints + self.n_wheels

    @property
    def n_legs(self) -> int:
        """Limbs on the torso (core and every segment) that have a servo joint
        and no wheel. A segment itself is a spine joint, not a limb."""
        kids: Dict[int, list] = {}
        for i, n in self.nodes.items():
            kids.setdefault(n.parent, []).append(i)

        def kinds(i):
            yield self.nodes[i].kind
            for j in kids.get(i, []):
                yield from kinds(j)

        limbs = []
        for i, n in self.nodes.items():
            if n.kind not in ("core", "segment"):
                continue
            for c in kids.get(i, []):
                if self.nodes[c].kind != "segment":
                    limbs.append(set(kinds(c)))
        return sum(any(is_joint(k) for k in s) and "wheel" not in s
                   for s in limbs)

    def layout(self) -> ActuatorLayout:
        out = []
        for i in self.order:
            n = self.nodes[i]
            if is_joint(n.kind):
                out.append(Actuator(i, "joint", n.axis))
            elif n.kind == "wheel":
                out.append(Actuator(i, "wheel", "wheel"))
        return out

    # geometry (rest pose) ------------------------------------------------
    def torso_polygon(self) -> List[Tuple[float, float]]:
        """The 8 contour vertices in the torso frame, counter-clockwise from
        the front: front midpoint, front-left corner, left midpoint, and so on.
        A contour of all ones is the old body_length x body_width rectangle."""
        l, w = self.body_length / 2, self.body_width / 2
        box = [(l, 0.0), (l, w), (0.0, w), (-l, w),
               (-l, 0.0), (-l, -w), (0.0, -w), (l, -w)]
        return [(float(factor * x), float(factor * y))
                for factor, (x, y) in zip(self.contour, box)]

    def segment_polygon(self, node: Node) -> List[Tuple[float, float]]:
        """The 8 contour vertices of a segment plate in its own frame, with
        the core numbering and all contour factors 1."""
        l, w = node.length / 2, node.width / 2
        box = [(l, 0.0), (l, w), (0.0, w), (-l, w),
               (-l, 0.0), (-l, -w), (0.0, -w), (l, -w)]
        return [(float(x), float(y)) for x, y in box]

    def slot_frame(self, parent: Node, slot: int,
                   child: Optional[Node] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Offset and rotation of a slot in the frame of its parent. The
        child's offset and angle place it along the slot; tips ignore them."""
        offset = 0.5 if child is None else child.offset
        angle = 0.5 if child is None else child.angle
        yaw = (2.0 * angle - 1.0) * np.radians(60.0)
        if parent.kind == "core" and slot in FACE_SLOTS:
            # Face mount: the child's X points straight up (top) or down
            # (bottom), then ANGLE yaws it about the vertical. The take-X-to-
            # +-Z turn is a rotation about Y, which keeps local Y horizontal,
            # and local Z horizontal too, so the joint frame stays defined.
            side, up = FACE_SLOTS[slot]
            poly = self.torso_polygon()
            x = FACE_X * poly[0 if side > 0 else 4][0]
            y = (2.0 * offset - 1.0) * FACE_Y * _half_width(poly, x)
            turn = _ry(-up * np.pi / 2)
            return (np.array([x, y, up * TORSO_HEIGHT / 2]),
                    _rz((2.0 * angle - 1.0) * np.pi) @ turn)
        if parent.kind in ("core", "segment"):
            if parent.kind == "core":
                poly, base = self.torso_polygon(), np.zeros(3)
            else:
                poly, base = self.segment_polygon(parent), segment_center(parent)
            k = CORE_VERTEX[slot]
            t = (2.0 * offset - 1.0) * 0.3
            a = poly[k]
            b = poly[(k + 1) % 8] if t >= 0 else poly[(k - 1) % 8]
            f = abs(t)
            x, y = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            return (base + np.array([x, y, 0]),
                    _rz(float(np.arctan2(y, x)) + yaw))
        if slot == 0:
            tip, rot = link_tip(parent)
            return tip, rot
        side = 1.0 if slot == 1 else -1.0
        tip, trot = link_tip(parent)
        d = elbow_distance(child) if child is not None else _scale(0.5, *ELBOW_RANGE)
        a = elbow_angle(child) if child is not None else _scale(
            0.5, np.radians(30.0), np.radians(150.0))
        return (tip + trot @ np.array([0, side * d, 0]), trot @ _rz(side * a))

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
                out.append((i, pos.copy(), foot_reach(n)))
            elif n.kind == "tail":
                out.append((i, pos + rot @ np.array([n.length, 0, 0]),
                            TAIL_TIP_RADIUS))
            elif n.kind == "skid":
                out.append((i, pos.copy(), SKID_RADIUS))
            elif n.kind == "spike":
                out.append((i, pos + rot @ np.array([SPIKE_LENGTH, 0, 0]),
                            SPIKE_RADIUS))
            elif n.kind == "segment":
                out.append((i, pos + rot @ segment_center(n),
                            SEGMENT_HEIGHT / 2))
            elif n.kind in ("joint", "link", "spring") and i not in cm:
                tip, _ = link_tip(n)
                out.append((i, pos + rot @ tip, section_radius(n)))
        return out

    @property
    def n_terminals(self) -> int:
        return len(self.terminal_points())

    @property
    def max_reach(self) -> float:
        """Largest distance from the torso centre to any point of the body."""
        r_xy = max(float(np.hypot(x, y)) for x, y in self.torso_polygon())
        r = float(np.hypot(r_xy, TORSO_HEIGHT / 2))
        for i, (pos, rot) in self.frames().items():
            n = self.nodes[i]
            r = max(r, float(np.linalg.norm(pos)))
            if n.kind in ("joint", "link", "spring"):
                for p in link_path(n):
                    r = max(r, float(np.linalg.norm(pos + rot @ p))
                            + section_radius(n))
            elif n.kind == "segment":
                centre, half = segment_box(n)
                corners = np.array([[sx, sy, sz] for sx in (-1, 1)
                                    for sy in (-1, 1) for sz in (-1, 1)]) * half
                r = max(r, max(float(np.linalg.norm(pos + rot @ (centre + c)))
                               for c in corners))
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
            if n.kind in ("joint", "wheel", "segment"):
                low = min(low, float((pos + corners @ rot.T)[:, 2].min()))
            if n.kind == "segment":
                centre, half = segment_box(n)
                box = np.array([[sx, sy, sz] for sx in (-1, 1)
                                for sy in (-1, 1) for sz in (-1, 1)]) * half
                low = min(low, float((pos + (centre + box) @ rot.T)[:, 2].min()))
            if n.kind in ("joint", "link", "spring"):
                r = section_radius(n)
                for p in link_path(n):
                    low = min(low, float((pos + rot @ p)[2]) - r)
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
                low = min(low, pos[2] + foot_lowest(n, rot))
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


def _inside(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Point-in-polygon test (ray casting) for a simple polygon."""
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def violations(morph: Morphology) -> List[str]:
    """Printability checks: link bending strength, bed footprint, elbow reach,
    and room for the electronics inside the torso contour."""
    out: List[str] = []
    poly = morph.torso_polygon()
    hx, hy = ELECTRONICS_BOX[0] / 2, ELECTRONICS_BOX[1] / 2
    probes = [(sx * hx, sy * hy) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    probes += [(sx * hx, 0.0) for sx in (-1.0, 1.0)]
    probes += [(0.0, sy * hy) for sy in (-1.0, 1.0)]
    if not all(_inside(x, y, poly) for x, y in probes):
        out.append("torso contour does not hold the electronics")
    for i, n in morph.nodes.items():
        if n.kind == "segment":
            extent = max(n.length, n.width)
            if extent > BED + 1e-9:
                out.append(f"#{i} segment: {extent * 1000:.0f} mm plate does "
                           f"not fit the {BED * 1000:.0f} mm bed")
            continue
        if n.kind not in ("joint", "link", "spring"):
            continue
        modulus = n.width * plate_thick(n) ** 2 / 6.0
        stress = SERVO_STALL / modulus if modulus > 0 else np.inf
        limit = PLA_STRENGTH * STRENGTH_SAFETY
        if stress > limit:
            out.append(f"#{i} {n.kind}: bending stress {stress / 1e6:.1f} MPa "
                       f"above the {limit / 1e6:.1f} MPa limit")
        pts = link_path(n)
        dx = max(p[0] for p in pts) - min(p[0] for p in pts) + n.width
        dy = max(p[1] for p in pts) - min(p[1] for p in pts) + n.width
        extent = max(dx, dy)
        if extent > BED + 1e-9:
            out.append(f"#{i} {n.kind}: {extent * 1000:.0f} mm "
                       f"({n.profile}) does not fit the "
                       f"{BED * 1000:.0f} mm bed")
        if n.slot in (1, 2) and morph.nodes[n.parent].kind in (
                "joint", "link", "spring"):
            parent_width = morph.nodes[n.parent].width
            if elbow_distance(n) + parent_width > BED + 1e-9:
                out.append(f"#{i} {n.kind}: elbow {elbow_distance(n) * 1000:.0f} + "
                           f"{parent_width * 1000:.0f} mm does not fit the "
                           f"{BED * 1000:.0f} mm bed")
    return out


def _segment_slot(kind: str, slot: int) -> bool:
    """True where a segment may attach: front or back of the core or a segment."""
    return kind in ("core", "segment") and slot in (0, 1)


def decode(genes: np.ndarray) -> Morphology:
    """Genome -> body. Total: any vector in [0, 1] gives a valid body."""
    genes = np.asarray(genes, dtype=float)
    assert genes.shape == (MORPH_LENGTH,), genes.shape
    blocks = genes.reshape(MAX_NODES, BLOCK_GENES)

    contour = tuple(round(_scale(blocks[0, g], *CONTOUR_RANGE), 6)
                    for g in CONTOUR_GENES)
    morph = Morphology(round(_scale(blocks[0, LENGTH], *BODY_BOUNDS[0]), 6),
                       round(_scale(blocks[0, WHEEL], *BODY_BOUNDS[1]), 6),
                       {0: Node(0, "core")}, contour)
    kinds = {0: "core"}
    for i in range(1, MAX_NODES):
        kinds[i] = TYPE_KINDS[_pick(blocks[i, TYPE], len(TYPE_KINDS))]

    # Who claims which slot: the lowest block id wins a clash. A segment only
    # claims the front or back of the core or of another segment, and no child
    # ever claims slot 1 of a segment (it faces the parent).
    claims: Dict[Tuple[int, int], int] = {}
    for i in range(1, MAX_NODES):
        b = blocks[i]
        if b[ACTIVE] <= 0.5:
            continue
        p = _pick(b[PARENT], MAX_NODES)
        if p == i or N_SLOTS[kinds[p]] == 0:
            continue
        s = _pick(b[SLOT], N_SLOTS[kinds[p]])
        if kinds[p] == "segment" and s == 1:
            continue
        if kinds[i] == "segment" and not _segment_slot(kinds[p], s):
            continue
        claims.setdefault((p, s), i)
    kids: Dict[int, List[Tuple[int, int]]] = {}
    for (p, s), i in claims.items():
        kids.setdefault(p, []).append((s, i))

    servos, segments = 0, 0
    frontier = [0]
    for _ in range(MAX_DEPTH):
        nxt = []
        for p in frontier:
            for s, i in sorted(kids.get(p, [])):
                kind = kinds[i]
                if is_joint(kind) or kind == "wheel":
                    if servos >= MAX_SERVOS:
                        continue
                    servos += 1
                if kind == "segment":
                    if segments >= MAX_SEGMENTS:
                        continue
                    segments += 1
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
        # a spring always bends about its blade's thin axis (local Y)
        n.axis = AXES[_pick(b[AXIS], 3)] if kind == "joint" else "yaw"
        n.flip = bool(b[FLIP] > 0.5)
    if kind == "segment":
        n.axis = AXES[_pick(b[AXIS], 3)]
        n.flip = bool(b[FLIP] > 0.5)
        n.length = round(_scale(b[LENGTH], *SEGMENT_LENGTH_RANGE), 6)
        n.width = round(_scale(b[WIDTH], *SEGMENT_WIDTH_RANGE), 6)
    if kind in ("joint", "link", "spring"):
        n.profile = PROFILES[_pick(b[PROFILE], len(PROFILES))]
        n.bend = round(float(np.radians(_scale(b[BEND], *BEND_RANGE))), 6)
    if kind in ("joint", "link", "spring", "tail"):
        n.length = round(_scale(b[LENGTH], *LENGTH_RANGE), 6)
    if kind == "wheel":
        n.wheel_rad = round(_scale(b[WHEEL], *WHEEL_RANGE_M), 6)
    if kind == "caster":
        n.wheel_rad = round(_scale(b[WHEEL], *CASTER_RANGE), 6)
    if kind == "foot":
        n.foot_rad = round(_scale(b[WHEEL], *FOOT_RANGE), 6)
        n.foot_shape = FOOT_SHAPES[_pick(b[AXIS], len(FOOT_SHAPES))]
    if kind == "tail":
        n.tail_mass = round(_scale(b[WHEEL], *TAIL_MASS_RANGE), 6)
    return n


def encode(morph: Morphology) -> np.ndarray:
    """Body -> canonical genome (inverse of decode)."""
    blocks = np.zeros((MAX_NODES, BLOCK_GENES))
    blocks[0, ACTIVE] = 1.0
    blocks[0, LENGTH] = _unscale(morph.body_length, *BODY_BOUNDS[0])
    blocks[0, WHEEL] = _unscale(morph.body_width, *BODY_BOUNDS[1])
    for k, g in enumerate(CONTOUR_GENES):
        blocks[0, g] = _unscale(morph.contour[k], *CONTOUR_RANGE)
    for i, n in morph.nodes.items():
        if i == 0:
            continue
        b = blocks[i]
        b[ACTIVE] = 1.0
        b[PARENT] = _bin(n.parent, MAX_NODES)
        b[SLOT] = _bin(n.slot, N_SLOTS[morph.nodes[n.parent].kind])
        b[ROT] = _unscale(n.rot, 0.0, 2 * np.pi)
        b[TYPE] = _bin(TYPE_KINDS.index(n.kind), len(TYPE_KINDS))
        b[AXIS] = (_bin(FOOT_SHAPES.index(n.foot_shape), len(FOOT_SHAPES))
                   if n.kind == "foot" else _bin(AXES.index(n.axis), 3))
        b[LENGTH] = _unscale(n.length, *(SEGMENT_LENGTH_RANGE
                                         if n.kind == "segment" else LENGTH_RANGE))
        if n.kind in ("joint", "link", "spring"):
            b[PROFILE] = _bin(PROFILES.index(n.profile), len(PROFILES))
            b[BEND] = _unscale(float(np.degrees(n.bend)), *BEND_RANGE)
        if n.kind == "foot":
            b[WHEEL] = _unscale(n.foot_rad, *FOOT_RANGE)
        elif n.kind == "tail":
            b[WHEEL] = _unscale(n.tail_mass, *TAIL_MASS_RANGE)
        elif n.kind == "caster":
            b[WHEEL] = _unscale(n.wheel_rad, *CASTER_RANGE)
        else:
            b[WHEEL] = _unscale(n.wheel_rad, *WHEEL_RANGE_M)
        b[FLIP] = 0.75 if n.flip else 0.25
        b[WIDTH] = _unscale(n.width, *(SEGMENT_WIDTH_RANGE
                                      if n.kind == "segment" else LINK_WIDTH_RANGE))
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


def _drop_subtree(morph: Morphology, root: int):
    stack = [root]
    while stack:
        k = stack.pop()
        stack += [c for c, n in morph.nodes.items() if n.parent == k]
        del morph.nodes[k]


def _ensure_servo(morph: Morphology):
    """A body without servos is a brick: hang a default joint on the core. If
    every core slot is taken, the first core child (and what hangs from it) is
    dropped; if every node id is used, one passive leaf is."""
    if morph.n_servos:
        return
    taken = morph.child_map().get(0, {})
    if len(taken) == N_SLOTS["core"]:
        _drop_subtree(morph, taken[min(taken)])
        taken = morph.child_map().get(0, {})
    if _free_id(morph) is None:
        parents = {n.parent for n in morph.nodes.values()}
        _drop_subtree(morph, max(k for k in morph.nodes if k != 0 and k not in parents))
        taken = morph.child_map().get(0, {})
    i = _free_id(morph)
    slot = next(s for s in (3, 2, 4, 6, 5, 7, 0, 1, 8, 9, 10, 11) if s not in taken)
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
            drop = [i for i in leaves if is_joint(morph.nodes[i].kind)
                    or morph.nodes[i].kind == "wheel"]
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
    if kind in ("core", "segment"):
        if slot >= 8:                   # face slots lie on the mid plane
            return slot
        return slot + 3 if 2 <= slot <= 4 else slot - 3 if slot >= 5 else slot
    return {1: 2, 2: 1}.get(slot, slot)


def copy_subtree(morph: Morphology, root: int, parent: int, slot: int,
                 mirror: bool = False,
                 source: Optional[Morphology] = None) -> Optional[int]:
    """
    Copy the subtree under `root` onto a free slot of `parent`, breadth first,
    stopping where a limit is hit (so a partial copy is still a valid body).
    mirror=True reflects it across the XZ plane: sides swapped, rolls
    negated, pitch and roll axes flipped. `source` is the body the subtree
    comes from (default: `morph` itself; another body for a graft). Returns
    the new root id or None.
    """
    source = morph if source is None else source
    if (parent, slot) in {(n.parent, n.slot) for n in morph.nodes.values()}:
        return None
    if morph.nodes[parent].kind == "segment" and slot == 1:
        return None
    base = morph.depth(parent) + 1 - source.depth(root)
    cm = source.child_map()
    queue = [(root, parent, slot)]
    new_root = None
    while queue:
        src, par, sl = queue.pop(0)
        node = source.nodes[src]
        i = _free_id(morph)
        if i is None or base + source.depth(src) > MAX_DEPTH:
            continue
        parent_kind = morph.nodes[par].kind
        if parent_kind == "segment" and sl == 1:
            continue
        if node.kind == "segment" and (
                not _segment_slot(parent_kind, sl)
                or morph.n_segments >= MAX_SEGMENTS):
            continue
        if (is_joint(node.kind) or node.kind == "wheel") \
                and morph.n_servos >= MAX_SERVOS:
            continue
        new = Node(**vars(node))
        new.id, new.parent, new.slot = i, par, sl
        if mirror:
            new.rot = (-node.rot) % (2 * np.pi)
            new.bend = -node.bend
            if par == 0 or morph.nodes[par].kind == "segment":
                # contour position and yaw mirror around the torso
                new.offset = round(1.0 - node.offset, 6)
                new.angle = round(1.0 - node.angle, 6)
            if node.kind in ("joint", "spring", "segment") \
                    and node.axis != "yaw":
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


def _random_contour(rng: np.random.Generator) -> Tuple[float, ...]:
    """Half the random bodies get a free contour, half the old rectangle."""
    if rng.random() < 0.5:
        return tuple(round(float(rng.uniform(*CONTOUR_RANGE)), 6) for _ in range(8))
    return (1.0,) * 8


def _random_seed(rng: np.random.Generator) -> Morphology:
    wheels = bool(rng.random() < 0.35)
    yaw = bool(rng.random() < 0.5)
    n_pairs = int(rng.integers(1, 4))
    while n_pairs > 1 and 2 * n_pairs * (2 + yaw + wheels) > MAX_SERVOS:
        n_pairs -= 1
    morph = template(n_pairs, wheels, yaw,
                     thigh=round(float(rng.uniform(*LENGTH_RANGE)), 6),
                     shin=round(float(rng.uniform(*LENGTH_RANGE)), 6),
                     wheel_rad=round(float(rng.uniform(0.03, 0.05)), 6),
                     body_length=round(float(rng.uniform(
                         0.30 if n_pairs > 1 else 0.15, 0.40)), 6),
                     body_width=round(float(rng.uniform(*BODY_BOUNDS[1])), 6))
    morph.contour = _random_contour(rng)
    return morph


# --------------------------------------------------------------------------
# Random bodies and mutation
# --------------------------------------------------------------------------

def _random_node(rng, kind: str, parent: int, slot: int) -> Node:
    n = Node(0, kind, parent, slot, float(rng.uniform(0, 2 * np.pi)))
    if kind in ("joint", "spring", "segment"):
        if kind == "joint":
            n.axis = str(rng.choice(AXES, p=[0.6, 0.25, 0.15]))
        elif kind == "segment":
            n.axis = str(rng.choice(AXES))
        else:
            n.axis = "yaw"
        n.flip = bool(rng.random() < 0.5)
    if kind in ("joint", "link", "spring", "tail"):
        n.length = round(float(rng.uniform(*LENGTH_RANGE)), 6)
    if kind == "segment":
        n.length = round(float(rng.uniform(*SEGMENT_LENGTH_RANGE)), 6)
        n.width = round(float(rng.uniform(*SEGMENT_WIDTH_RANGE)), 6)
    if kind in ("joint", "link", "spring"):
        n.profile = str(rng.choice(PROFILES, p=[0.6, 0.2, 0.2]))
        if n.profile != "straight":
            n.bend = round(float(np.radians(rng.uniform(*BEND_RANGE))), 6)
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
        n.foot_shape = str(rng.choice(FOOT_SHAPES))
    return n


def _add_random(morph: Morphology, rng, p: int, s: int) -> Optional[int]:
    i = _free_id(morph)
    if i is None or morph.depth(p) + 1 > MAX_DEPTH:
        return None
    kinds = tuple(k for k in TYPE_KINDS if k != "segment")
    kind = str(rng.choice(kinds,
                          p=[0.45, 0.14, 0.10, 0.18, 0.05, 0.03, 0.02,
                             0.02, 0.01]))
    if (is_joint(kind) or kind == "wheel") and morph.n_servos >= MAX_SERVOS:
        kind = "link" if rng.random() < 0.5 else "foot"
    node = _random_node(rng, kind, p, s)
    node.id = i
    morph.nodes[i] = node
    return i


def _add_segment(morph: Morphology, rng, p: int, s: int) -> Optional[int]:
    """A segment on a core or segment front/back slot, within the limit."""
    i = _free_id(morph)
    if (i is None or morph.n_segments >= MAX_SEGMENTS
            or not _segment_slot(morph.nodes[p].kind, s)
            or morph.depth(p) + 1 > MAX_DEPTH):
        return None
    node = _random_node(rng, "segment", p, s)
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
                       {0: Node(0, "core")}, _random_contour(rng))
    if rng.random() < 0.1:          # one spine segment in about one body in ten
        slots = [s for s in (0, 1) if s not in morph.child_map().get(0, {})]
        if slots:
            _add_segment(morph, rng, 0, slots[int(rng.integers(len(slots)))])
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


BIG_STEP_P = 0.2      # share of parameter steps that are large jumps
SMALL_STEP, BIG_STEP = 0.1, 0.4     # step sizes in normalised gene units


def _jitter(value: float, lo: float, hi: float, rng) -> float:
    """Gaussian step in normalised units, clipped to the gene range; one step
    in five is a large jump, so sizes can leave the neighbourhood they start in."""
    sigma = BIG_STEP if rng.random() < BIG_STEP_P else SMALL_STEP
    u = _unscale(value, lo, hi) + rng.normal(0.0, sigma)
    return round(_scale(float(np.clip(u, 0.0, 1.0)), lo, hi), 6)


def _op_param(morph, rng):
    n = morph.nodes[int(rng.choice(list(morph.nodes)))]
    if n.kind == "core":
        what = str(rng.choice(["length", "width", "contour"]))
        if what == "length":
            morph.body_length = _jitter(morph.body_length, *BODY_BOUNDS[0], rng)
        elif what == "width":
            morph.body_width = _jitter(morph.body_width, *BODY_BOUNDS[1], rng)
        else:
            k = int(rng.integers(len(morph.contour)))
            contour = list(morph.contour)
            contour[k] = _jitter(contour[k], *CONTOUR_RANGE, rng)
            morph.contour = tuple(contour)
        return
    choices = ["rot", "offset", "angle"]
    if n.kind in ("joint", "link", "spring"):
        choices += ["length"] * 2 + ["width", "thick", "profile"]
        if n.profile != "straight":
            choices += ["bend"]
    if n.kind == "segment":
        choices += ["length"] * 2 + ["width", "axis", "flip"]
    if n.kind == "tail":
        choices += ["length"] * 2 + ["tail_mass"]
    if n.kind == "joint":
        choices += ["axis", "flip"]
    if n.kind == "spring":
        choices += ["flip"]       # the axis is fixed: see _node_from_block
    if n.kind in ("wheel", "caster"):
        choices += ["wheel_rad"] * 2
    if n.kind == "foot":
        choices += ["foot_rad", "foot_shape"]
    what = str(rng.choice(choices))
    if what == "rot":
        n.rot = _jitter(n.rot, 0.0, 2 * np.pi, rng) % (2 * np.pi)
    elif what == "length":
        span = SEGMENT_LENGTH_RANGE if n.kind == "segment" else LENGTH_RANGE
        n.length = _jitter(n.length, *span, rng)
    elif what == "width":
        span = SEGMENT_WIDTH_RANGE if n.kind == "segment" else LINK_WIDTH_RANGE
        n.width = _jitter(n.width, *span, rng)
    elif what == "thick":
        n.thick = _jitter(n.thick, *(SPRING_BLADE_RANGE if n.kind == "spring"
                                     else LINK_THICK_RANGE), rng)
    elif what == "offset":
        n.offset = _jitter(n.offset, 0.0, 1.0, rng)
    elif what == "angle":
        n.angle = _jitter(n.angle, 0.0, 1.0, rng)
    elif what == "profile":
        n.profile = str(rng.choice(PROFILES))
        if n.profile == "straight":
            n.bend = 0.0
        elif not n.bend:
            n.bend = round(float(np.radians(rng.uniform(*BEND_RANGE))), 6)
    elif what == "bend":
        n.bend = round(float(np.radians(_jitter(
            float(np.degrees(n.bend)), *BEND_RANGE, rng))), 6)
    elif what == "axis":
        n.axis = str(rng.choice(AXES))
    elif what == "flip":
        n.flip = not n.flip
    elif what == "wheel_rad":
        span = CASTER_RANGE if n.kind == "caster" else WHEEL_RANGE_M
        n.wheel_rad = _jitter(n.wheel_rad, *span, rng)
    elif what == "tail_mass":
        n.tail_mass = _jitter(n.tail_mass, *TAIL_MASS_RANGE, rng)
    elif what == "foot_shape":
        n.foot_shape = str(rng.choice(FOOT_SHAPES))
    else:
        n.foot_rad = _jitter(n.foot_rad, *FOOT_RANGE, rng)


def _op_type(morph, rng):
    """Swap the type of a limb segment, or of a leaf (also wheel and foot)."""
    parents = {n.parent for n in morph.nodes.values()}
    cand = [i for i, n in morph.nodes.items() if i != 0
            and (n.kind in ("joint", "link", "spring", "segment")
                 or i not in parents)]
    if not cand:
        return
    i = int(rng.choice(cand))
    n = morph.nodes[i]
    kinds = ["joint", "link", "spring"] + (
        ["wheel", "foot", "tail", "skid", "spike", "caster"]
        if i not in parents else [])
    if n.kind == "segment":
        kinds.append("segment")
    elif (morph.n_segments < MAX_SEGMENTS
          and _segment_slot(morph.nodes[n.parent].kind, n.slot)):
        kinds.append("segment")
    kind = str(rng.choice([k for k in kinds if k != n.kind]))
    if (is_joint(kind) or kind == "wheel") \
            and not (is_joint(n.kind) or n.kind == "wheel") \
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
    kids = morph.child_map().get(i, {})
    bad = [c for s, c in kids.items()
           if (s == 1 if kind == "segment" else s >= N_SLOTS[kind])]
    for c in bad:
        _drop_subtree(morph, c)


def _op_add(morph, rng):
    """A new module, sometimes with one more behind it, in a free slot. About
    one add in ten is a spine segment when a front or back slot is free."""
    free = [fs for fs in morph.free_slots() if morph.depth(fs[0]) < MAX_DEPTH]
    if not free:
        return
    seg = [fs for fs in free
           if _segment_slot(morph.nodes[fs[0]].kind, fs[1])]
    if seg and morph.n_segments < MAX_SEGMENTS and rng.random() < 0.1:
        _add_segment(morph, rng, *seg[int(rng.integers(len(seg)))])
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


def _op_scale(morph, rng):
    """Stretch or shrink a whole limb: every length in a subtree times one
    factor between 1/2 and 2, so long-legged and stubby bodies are one step
    away instead of many."""
    cand = [i for i in morph.nodes if i != 0]
    if not cand:
        return
    factor = float(np.exp(rng.uniform(-np.log(2), np.log(2))))
    cm = morph.child_map()
    stack = [int(rng.choice(cand))]
    while stack:
        n = morph.nodes[stack.pop()]
        stack.extend(cm.get(n.id, {}).values())
        if n.kind == "segment":
            n.length = round(float(np.clip(
                n.length * factor, *SEGMENT_LENGTH_RANGE)), 6)
        elif n.kind in ("joint", "link", "spring", "tail"):
            n.length = round(float(np.clip(n.length * factor, *LENGTH_RANGE)), 6)


MUTATIONS = ((_op_param, 0.30), (_op_type, 0.15), (_op_add, 0.2),
             (_op_remove, 0.1), (_op_mirror, 0.15), (_op_copy, 0.05),
             (_op_scale, 0.05))


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


def graft(genes: np.ndarray, donor: np.ndarray,
          rng: np.random.Generator) -> np.ndarray:
    """
    Crossover between two bodies: a limb of `donor` (a whole subtree) is copied
    onto a free slot of `genes`, mirrored onto the opposite torso side as well
    when that side is free, as in Sims (1994). Cutting two trees in half and
    swapping the halves would mostly give broken bodies; a graft keeps both
    pieces whole. Returns `genes` unchanged when no limb or slot fits.
    """
    morph, other = decode(genes), decode(donor)
    limbs = [i for i in other.nodes if i != 0]
    free = [fs for fs in morph.free_slots() if morph.depth(fs[0]) < MAX_DEPTH]
    if not limbs or not free:
        return encode(morph)
    root = int(rng.choice(limbs))
    p, s = free[int(rng.integers(len(free)))]
    new = copy_subtree(morph, root, p, s, source=other)
    if new is not None and p == 0 and rng.random() < 0.5:
        ms = mirror_slot("core", s)
        if ms != s:
            copy_subtree(morph, new, 0, ms, mirror=True)
    return encode(prune(morph))


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def summary(m: Morphology) -> str:
    poly = m.torso_polygon()
    size_x = max(x for x, _ in poly) - min(x for x, _ in poly)
    size_y = max(y for _, y in poly) - min(y for _, y in poly)
    contour = " ".join(f"{c:.2f}" for c in m.contour)
    lines = [f"torso {size_x:.2f} x {size_y:.2f} m, contour {contour}, "
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
            elif n.kind == "segment":
                desc += (f" {n.axis}{'-' if n.flip else ''} "
                         f"{n.length * 1000:.0f}x{n.width * 1000:.0f}mm")
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

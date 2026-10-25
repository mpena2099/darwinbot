"""
parts.py
--------
Parametric printable parts (build123d) for the real DarwinBot: servo bracket,
link, elbow plate, TPU foot, wheel and torso plate (one mount per used core slot). All dimensions in mm.

Conventions: a link lies along +X with the horn axis at the origin and the
next servo axis at x = length_mm; servo and wheel axes point along Z.
Minimum wall 2 mm; slots and holes are vertical, so little needs support.
"""

import math

from build123d import (Align, Box, BuildPart, Circle, Cylinder, Locations, Part,
                       Pos, Rot, Sphere, SlotOverall, extrude, export_stl as _export_stl,
                       fillet)

# --- measure a real STS3215 and update ------------------------------------
SERVO_L = 45.2          # body length (along the output axis offset direction)
SERVO_W = 24.7          # body width
SERVO_H = 35.0          # body height
SERVO_CLEARANCE = 0.4   # printing slack per side
HORN_BOLT_CIRCLE = 20.0 # PLACEHOLDER: horn screw pattern diameter
HORN_N_HOLES = 4        # PLACEHOLDER: number of horn screw holes
HORN_HOLE_D = 3.0       # PLACEHOLDER: horn screw hole diameter
HORN_BORE_D = 6.0       # PLACEHOLDER: clearance for the horn hub
EAR_HOLE_D = 3.4        # PLACEHOLDER: servo mounting ear screw hole
EAR_HOLE_DX = 40.0      # PLACEHOLDER: ear hole spacing along the body
# ---------------------------------------------------------------------------

WALL = 2.5              # wall thickness (>= 2)
PLATE_T = 4.0           # torso plate thickness
LINK_T = 6.0            # default link / elbow plate thickness
LINK_W = 20.0           # default link width = end boss diameter
M3 = 3.2                # M3 clearance hole
BED_MM = 220.0          # largest part that fits a common print bed
SPLICE_W = 24.0         # splice strip width, across the cut
SPLICE_EDGE = 12.0      # splice hole distance from the strip ends
SPLICE_HOLE = 8.0       # splice hole distance from the cut on each side
WHEEL_T = 10.0          # wheel thickness
FOOT_R = 9.0            # default foot cap radius
FOOT_SOCKET_D = 12.0    # PLACEHOLDER: shin tube/socket diameter
FOOT_MIN_R = 8.6        # keeps the wall around the socket >= 2 mm
SPRING_FLEX = 12.0      # TPU blade length, matches morphology.SPRING_FLEX_LEN
SPRING_BLOCK = 10.0     # mounting block length at each end of the blade


def _horn_holes(thickness: float) -> Part:
    """Cutter: horn bore plus the screw circle, centred on the origin."""
    with BuildPart() as c:
        Cylinder(HORN_BORE_D / 2, thickness * 3)
        for i in range(HORN_N_HOLES):
            a = 2 * math.pi * i / HORN_N_HOLES
            with Locations((HORN_BOLT_CIRCLE / 2 * math.cos(a), HORN_BOLT_CIRCLE / 2 * math.sin(a))):
                Cylinder(HORN_HOLE_D / 2, thickness * 3)
    return c.part


def servo_bracket() -> Part:
    """Open-top box holding one servo body, with ear screw holes."""
    iw, il = SERVO_W + 2 * SERVO_CLEARANCE, SERVO_L + 2 * SERVO_CLEARANCE
    ow, ol = iw + 2 * WALL, il + 2 * WALL
    h = SERVO_H * 0.6 + WALL            # holds the lower 60% of the body
    box = Box(ol, ow, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    pocket = Pos(0, 0, WALL) * Box(il, iw, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = box - pocket
    # ear screws go through the floor, along the servo length
    for sx in (-EAR_HOLE_DX / 2, EAR_HOLE_DX / 2):
        part -= Pos(sx, 0, 0) * Cylinder(EAR_HOLE_D / 2, WALL * 3)
    # two M3 holes through each side wall for cross screws
    for sx in (-il / 4, il / 4):
        part -= Pos(sx, 0, h / 2 + WALL) * Rot(X=90) * Cylinder(M3 / 2, ow * 2)
    return part


def link(length_mm: float, width_mm: float = LINK_W,
         thick_mm: float = LINK_T) -> Part:
    """Plate between two servo axes (horn at x=0, next bracket at x=length)."""
    if length_mm < width_mm:
        raise ValueError("link too short")
    body = Pos(length_mm / 2, 0, 0) * (
        Box(length_mm + width_mm, width_mm, thick_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN)))
    part = fillet(body.edges().filter_by(lambda e: abs(e.length - thick_mm) < 1e-6),
                  width_mm / 2 - 0.01)
    part -= _horn_holes(thick_mm)
    # bracket end: pair of M3 holes plus a central slot for the cable
    for dy in (-6, 6):
        part -= Pos(length_mm, dy, 0) * Cylinder(M3 / 2, thick_mm * 3)
    part -= Pos(length_mm, 0, 0) * Cylinder(HORN_BORE_D / 2, thick_mm * 3)
    return part


def _link_end_pattern(thick_mm: float) -> Part:
    """Cutter at the origin: cable bore plus the bracket end M3 pair."""
    cut = Cylinder(HORN_BORE_D / 2, thick_mm * 3)
    for dy in (-6, 6):
        cut += Pos(0, dy, 0) * Cylinder(M3 / 2, thick_mm * 3)
    return cut


def _link_end_holes(x: float, y: float, angle_deg: float = 0.0,
                    thick_mm: float = PLATE_T) -> Part:
    """The bracket end pattern placed at (x, y) and turned about Z."""
    return Pos(x, y, 0) * Rot(Z=angle_deg) * _link_end_pattern(thick_mm)


def elbow_plate(distance_mm: float, angle_deg: float = 90.0,
                width_mm: float = LINK_W, thick_mm: float = LINK_T) -> Part:
    """Bar bolted on a link end (origin) that carries a side module
    `distance_mm` away along +Y; the far end repeats the link end pattern,
    turned about Z by angle_deg - 90."""
    if distance_mm < width_mm:
        raise ValueError("elbow too short")
    body = Pos(0, distance_mm / 2, 0) * Box(
        width_mm, distance_mm + width_mm, thick_mm,
        align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = fillet(body.edges().filter_by(lambda e: abs(e.length - thick_mm) < 1e-6),
                  width_mm / 2 - 0.01)
    part -= _link_end_holes(0, 0, 0.0, thick_mm)
    part -= _link_end_holes(0, distance_mm, angle_deg - 90.0, thick_mm)
    return part


def spring(length_mm: float, width_mm: float = LINK_W,
           blade_mm: float = 4.0) -> Part:
    """TPU flexure (print in TPU): two mounting blocks with the link end
    pattern, joined by a blade of blade_mm thickness and SPRING_FLEX length.
    The blade lies flat, so it bends about Y; turn the part to get the axis of
    the spring. length_mm is the link it sits under; the module link is a
    separate PLA plate on the far block, so only block, blade and block are
    printed here."""
    if length_mm < SPRING_FLEX + 2 * SPRING_BLOCK:
        raise ValueError("spring too short")
    t = LINK_T
    total = SPRING_FLEX + 2 * SPRING_BLOCK
    part = Box(total, width_mm, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    if blade_mm < t:                   # thin the middle down to the blade
        part -= (Pos(SPRING_BLOCK, 0, t / 2)
                 * Box(SPRING_FLEX, width_mm, t - blade_mm,
                       align=(Align.CENTER, Align.CENTER, Align.CENTER)))
    for x in (SPRING_BLOCK / 2, total - SPRING_BLOCK / 2):
        part -= _link_end_holes(x, 0, 0.0, t)
    return part


def foot(radius_mm: float = FOOT_R) -> Part:
    """Rounded cap (print in TPU): hemisphere over a socket for the shin.
    Scaled from the 9 mm foot; the socket keeps its diameter and the radius is
    clamped so the wall around it stays at least 2 mm."""
    r = max(radius_mm, FOOT_MIN_R)
    with BuildPart() as b:
        Sphere(r)
    dome = b.part - Pos(0, 0, -r) * Box(4 * r, 4 * r, 2 * r)
    stem = Pos(0, 0, 0) * Cylinder(r, 10 * r / FOOT_R,
                                   align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = dome + stem
    part -= Pos(0, 0, 3) * Cylinder(FOOT_SOCKET_D / 2, 20,
                                    align=(Align.CENTER, Align.CENTER, Align.MIN))
    return part


def wheel(radius_mm: float) -> Part:
    """Disc on a servo horn; V groove on the rim holds an O-ring tyre."""
    if not 30 <= radius_mm <= 60:
        raise ValueError("wheel radius must be 30..60 mm")
    disc = Cylinder(radius_mm, WHEEL_T)
    # groove: 3 mm deep, 4 mm wide, rectangular, at mid-thickness
    ring = Cylinder(radius_mm + 1, 4) - Cylinder(radius_mm - 3, 4 + 1)
    part = disc - ring
    part -= _horn_holes(WHEEL_T)
    # lightening holes keep walls >= 2 mm: only when the disc is big enough
    if radius_mm >= 40:
        r_mid = (radius_mm - 3 + HORN_BOLT_CIRCLE / 2 + HORN_HOLE_D) / 2
        d = min(10.0, (radius_mm - 3 - HORN_BOLT_CIRCLE / 2 - HORN_HOLE_D) - 2 * WALL)
        if d > 4:
            for i in range(6):
                a = 2 * math.pi * (i + 0.5) / 6
                part -= Pos(r_mid * math.cos(a), r_mid * math.sin(a), 0) * Cylinder(d / 2, WHEEL_T * 3)
    return part


def torso_plate(length_mm: float, width_mm: float, mounts) -> Part:
    """Base plate; each used core slot, given as an (x, y) mm position in `mounts`,
    gets a tab under its servo bracket and 4 M3 holes. A mount on a side edge
    holds a servo pointing along Y, any other one a servo along X. The bracket
    is centred on the slot as in the simulation, so half of it overhangs the
    edge and the tab carries it."""
    t = PLATE_T
    part = Box(length_mm, width_mm, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = fillet(part.edges().filter_by(lambda e: abs(e.length - t) < 1e-6), 5)
    ol = SERVO_L + 2 * SERVO_CLEARANCE + 2 * WALL
    ow = SERVO_W + 2 * SERVO_CLEARANCE + 2 * WALL
    feet = []                                   # bracket footprints (x, y, size x, size y)
    for x, y in mounts:
        side = abs(abs(y) - width_mm / 2) < 1e-3
        fx, fy = (ow, ol) if side else (ol, ow)
        feet.append((x, y, fx, fy))
        part += Pos(x, y, 0) * Box(fx, fy, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for x, y, fx, fy in feet:
        for sx in (-1, 1):
            for sy in (-1, 1):
                part -= Pos(x + sx * (fx / 2 - 3), y + sy * (fy / 2 - 3), 0) * Cylinder(M3 / 2, t * 3)
    # ESP32 board: 58 x 23 mm hole pattern, centred behind the origin
    for sx in (-29, 29):
        for sy in (-11.5, 11.5):
            part -= Pos(sx, sy, 0) * Cylinder(1.6, t * 3)
    # battery strap slot across the plate, as far back as the brackets allow
    strap_w = min(width_mm - 20, 30)
    def free(sx):
        return all(abs(sx - x) > fx / 2 + 4 or abs(y) > fy / 2 + strap_w / 2 + 2
                   for x, y, fx, fy in feet)
    xs = [-length_mm / 2 + 12 + i for i in range(int(length_mm / 2 - 12 - 35))]
    strap = next((sx for sx in xs if free(sx)), None)
    if strap is not None:
        part -= Pos(strap, 0, 0) * Box(4, strap_w, t * 3)
    return part


def _plate_bounds(length_mm: float, width_mm: float, mounts):
    """Bounding box of torso_plate in mm, brackets included."""
    x0, x1 = -length_mm / 2, length_mm / 2
    y0, y1 = -width_mm / 2, width_mm / 2
    ol = SERVO_L + 2 * SERVO_CLEARANCE + 2 * WALL
    ow = SERVO_W + 2 * SERVO_CLEARANCE + 2 * WALL
    for x, y in mounts:
        side = abs(abs(y) - width_mm / 2) < 1e-3
        fx, fy = (ow, ol) if side else (ol, ow)
        x0, x1 = min(x0, x - fx / 2), max(x1, x + fx / 2)
        y0, y1 = min(y0, y - fy / 2), max(y1, y + fy / 2)
    return x0, x1, y0, y1


def tile_plan(length_mm: float, width_mm: float, mounts):
    """Fewest equal whole-millimetre tiles of the torso that fit BED_MM:
    (x0, y0, tile width, tile height, nx, ny) in mm."""
    x0, x1, y0, y1 = _plate_bounds(length_mm, width_mm, mounts)
    nx = max(1, math.ceil((x1 - x0) / BED_MM))
    ny = max(1, math.ceil((y1 - y0) / BED_MM))
    return (x0, y0, math.ceil((x1 - x0) / nx), math.ceil((y1 - y0) / ny), nx, ny)


def splice_plate(span_mm: float, width_mm: float = SPLICE_W,
                 thick_mm: float = PLATE_T) -> Part:
    """Strip laid across a torso cut, with the four M3 holes drilled in the
    tiles: two on each side of the cut, SPLICE_EDGE from the strip ends."""
    if span_mm < 2 * SPLICE_EDGE:
        raise ValueError("splice too short")
    body = Box(width_mm, span_mm, thick_mm,
               align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = body
    for sx in (-SPLICE_HOLE, SPLICE_HOLE):
        for sy in (-1, 1):
            part -= Pos(sx, sy * (span_mm / 2 - SPLICE_EDGE), 0) * \
                Cylinder(M3 / 2, thick_mm * 3)
    return part


def torso_tiles(length_mm: float, width_mm: float, mounts) -> list[Part]:
    """Printable torso pieces: torso_plate when it fits BED_MM, else the
    fewest equal tiles (grid along X and Y), each cut drilled for a splice."""
    plate = torso_plate(length_mm, width_mm, mounts)
    x0, y0, tw, th, nx, ny = tile_plan(length_mm, width_mm, mounts)
    if nx == 1 and ny == 1:
        return [plate]
    holes = []
    for i in range(1, nx):                      # vertical cuts
        for j in range(ny):
            ya = y0 + j * th + SPLICE_EDGE
            yb = y0 + (j + 1) * th - SPLICE_EDGE
            x = x0 + i * tw
            holes += [(x - SPLICE_HOLE, ya), (x + SPLICE_HOLE, ya),
                      (x - SPLICE_HOLE, yb), (x + SPLICE_HOLE, yb)]
    for j in range(1, ny):                      # horizontal cuts
        for i in range(nx):
            xa = x0 + i * tw + SPLICE_EDGE
            xb = x0 + (i + 1) * tw - SPLICE_EDGE
            y = y0 + j * th
            holes += [(xa, y - SPLICE_HOLE), (xa, y + SPLICE_HOLE),
                      (xb, y - SPLICE_HOLE), (xb, y + SPLICE_HOLE)]
    tiles = []
    for j in range(ny):
        for i in range(nx):
            cell = Pos(x0 + (i + 0.5) * tw, y0 + (j + 0.5) * th, 0) * Box(
                tw, th, 3 * PLATE_T, align=(Align.CENTER, Align.CENTER, Align.CENTER))
            tile = plate & cell
            for x, y in holes:
                tile -= Pos(x, y, 0) * Cylinder(M3 / 2, PLATE_T * 3)
            tiles.append(tile)
    return tiles


def export_stl(part: Part, path) -> None:
    """Write part as a binary STL."""
    _export_stl(part, str(path))

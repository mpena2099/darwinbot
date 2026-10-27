"""
parts.py
--------
Parametric printable parts (build123d) for the real DarwinBot: servo bracket,
link, elbow plate, TPU foot, wheel and torso plate (the extruded contour
polygon, one mount per used core slot). All dimensions in mm.

Conventions: a link lies along +X with the horn axis at the origin and the
next servo axis at x = length_mm; servo and wheel axes point along Z.
Minimum wall 2 mm; slots and holes are vertical, so little needs support.
"""

import math

from build123d import (Align, Box, BuildPart, Circle, Cone, Cylinder, Locations,
                       Part, Polygon, Pos, Rot, Sphere, SlotOverall, extrude,
                       export_stl as _export_stl)

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
PLATE_T = 6.0           # torso plate thickness (its tabs carry the edge servos)
LINK_T = 6.0            # default link / elbow plate thickness: 4 mm bends at ~36 MPa under the stall torque, 6 mm at ~16
LINK_W = 20.0           # default link width = end boss diameter
M3 = 3.2                # M3 clearance hole
BED_MM = 220.0          # largest part that fits a common print bed
SPLICE_W = 24.0         # splice strip width, across the cut
SPLICE_EDGE = 12.0      # splice hole distance from the strip ends
SPLICE_HOLE = 8.0       # splice hole distance from the cut on each side
WHEEL_T = 10.0          # wheel thickness
CASTER_T = 8.0          # caster wheel thickness, matches morphology.CASTER_WHEEL_T
CASTER_GAP = CASTER_T + 1.2   # clearance between the fork arms
CASTER_ARM = 7.5        # fork arm width, keeps >= 2 mm around the axle hole
FORK_T = 10.0          # fork thickness: keeps the axle bore 3 mm clear of the faces
CASTER_L = 44.0         # fork length along the arms
FOOT_R = 9.0            # default foot cap radius
FOOT_SOCKET_D = 12.0    # PLACEHOLDER: shin tube/socket diameter
FOOT_MIN_R = 8.6        # keeps the wall around the socket >= 2 mm
SPRING_FLEX = 12.0      # TPU blade length, matches morphology.SPRING_FLEX_LEN
MOUNT_SIDE = 16.0       # M3 mounting pattern: 4 holes on a square of this side
MOUNT_MIN_W = MOUNT_SIDE + M3 + 2 * WALL   # narrowest part that holds the pattern
END_BOSS_D = HORN_BOLT_CIRCLE + 2 * (HORN_HOLE_D / 2 + WALL)   # keeps the horn holes inside
SPRING_BLOCK = MOUNT_MIN_W   # mounting block length at each end of the blade
TAIL_W = 15.0           # tail bar width, matches morphology.TAIL_BAR_SIZE
TAIL_T = 6.0            # tail bar thickness
SKID_L, SKID_W, SKID_T = 60.0, 30.0, 6.0   # low-friction pad
SPIKE_L, SPIKE_R = 30.0, 5.0               # TPU tip, tapered to SPIKE_TIP_R
SPIKE_TIP_R = 1.5


def _horn_holes(thickness: float) -> Part:
    """Cutter: horn bore plus the screw circle, centred on the origin."""
    with BuildPart() as c:
        Cylinder(HORN_BORE_D / 2, thickness * 3)
        for i in range(HORN_N_HOLES):
            a = 2 * math.pi * i / HORN_N_HOLES
            with Locations((HORN_BOLT_CIRCLE / 2 * math.cos(a), HORN_BOLT_CIRCLE / 2 * math.sin(a))):
                Cylinder(HORN_HOLE_D / 2, thickness * 3)
    return c.part


def _mount_holes(thickness: float, yaw_deg: float = 0.0) -> Part:
    """Cutter at the origin: the common M3 mounting pattern (4 holes on a
    MOUNT_SIDE square), turned about Z. Every mounting interface uses it: the
    servo bracket floor, link ends, elbow plate ends, spring blocks and torso
    tabs, so any of them bolts to any other."""
    h = MOUNT_SIDE / 2
    cut = None
    for sx in (-h, h):
        for sy in (-h, h):
            hole = Pos(sx, sy, 0) * Cylinder(M3 / 2, thickness * 3)
            cut = hole if cut is None else cut + hole
    return Rot(Z=yaw_deg) * cut


def _boss_d(width_mm: float) -> float:
    """Diameter of a link end boss: wide enough for the horn holes."""
    return max(width_mm, END_BOSS_D)


def _bar(x0: float, x1: float, width_mm: float, thick_mm: float) -> Part:
    """Flat bar along X from x0 to x1 with a round boss at each end."""
    d = _boss_d(width_mm)
    bar = Pos((x0 + x1) / 2, 0, 0) * Box(
        x1 - x0, width_mm, thick_mm, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for x in (x0, x1):
        bar += Pos(x, 0, 0) * Cylinder(d / 2, thick_mm,
                                       align=(Align.CENTER, Align.CENTER, Align.MIN))
    return bar


def servo_bracket() -> Part:
    """Open-top box holding one servo body, with ear screw holes."""
    iw, il = SERVO_W + 2 * SERVO_CLEARANCE, SERVO_L + 2 * SERVO_CLEARANCE
    ow, ol = iw + 2 * WALL, il + 2 * WALL
    h = SERVO_H * 0.6 + WALL            # holds the lower 60% of the body
    box = Box(ol, ow, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    pocket = Pos(0, 0, WALL) * Box(il, iw, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = box - pocket
    # mounting pattern through the floor, centred under the servo axis
    part -= Pos(0, 0, 0) * _mount_holes(WALL)
    # ear screws go through the floor, along the servo length
    for sx in (-EAR_HOLE_DX / 2, EAR_HOLE_DX / 2):
        part -= Pos(sx, 0, 0) * Cylinder(EAR_HOLE_D / 2, WALL * 3)
    # two M3 holes through each side wall for cross screws
    for sx in (-il / 4, il / 4):
        part -= Pos(sx, 0, h / 2 + WALL) * Rot(X=90) * Cylinder(M3 / 2, ow * 2)
    return part


def link(length_mm: float, width_mm: float = LINK_W,
         thick_mm: float = LINK_T) -> Part:
    """Plate between two servo axes (horn at x=0, next bracket at x=length).
    Both ends carry a round boss of END_BOSS_D (or the width, if wider), so the
    horn holes stay inside the material for any width."""
    if length_mm < width_mm:
        raise ValueError("link too short")
    part = _bar(0.0, length_mm, width_mm, thick_mm)
    part -= _horn_holes(thick_mm)
    # bracket end: the common mounting pattern plus a bore for the cable
    part -= Pos(length_mm, 0, 0) * _link_end_pattern(thick_mm)
    return part


def _link_end_pattern(thick_mm: float, yaw_deg: float = 0.0) -> Part:
    """Cutter at the origin: cable bore plus the mounting pattern."""
    return Cylinder(HORN_BORE_D / 2, thick_mm * 3) + _mount_holes(thick_mm, yaw_deg)


def _link_end_holes(x: float, y: float, angle_deg: float = 0.0,
                    thick_mm: float = PLATE_T) -> Part:
    """The bracket end pattern placed at (x, y) and turned about Z."""
    return Pos(x, y, 0) * _link_end_pattern(thick_mm, angle_deg)


def elbow_plate(distance_mm: float, angle_deg: float = 90.0,
                width_mm: float = LINK_W, thick_mm: float = LINK_T) -> Part:
    """Bar bolted on a link end (origin) that carries a side module
    `distance_mm` away along +Y; the far end repeats the link end pattern,
    turned about Z by angle_deg - 90."""
    if distance_mm < width_mm:
        raise ValueError("elbow too short")
    part = Rot(Z=90) * _bar(0.0, distance_mm, width_mm, thick_mm)
    part -= _link_end_holes(0, 0, 0.0, thick_mm)
    part -= _link_end_holes(0, distance_mm, angle_deg - 90.0, thick_mm)
    return part


def spring(width_mm: float = LINK_W, blade_mm: float = 4.0) -> Part:
    """TPU flexure (print in TPU): block, blade, block along X. Each block
    carries the common mounting pattern; the blade is exactly blade_mm thick
    (2..8 mm), SPRING_FLEX long and as wide as the part. The blade lies flat,
    so it bends about Y; turn the part to get the axis of the spring. The PLA
    link of the module is a separate plate bolted on a block, so the part does
    not depend on the link length. Blocks are at least MOUNT_MIN_W wide to
    hold the pattern, and as thick as the blade when it is thicker than a link."""
    if not 2.0 <= blade_mm <= 8.0:
        raise ValueError("spring blade must be 2..8 mm")
    t = max(LINK_T, blade_mm)
    wb = max(width_mm, MOUNT_MIN_W)
    total = SPRING_FLEX + 2 * SPRING_BLOCK
    xb = (SPRING_FLEX + SPRING_BLOCK) / 2          # block centres, at +-xb
    part = Pos(-xb, 0, 0) * Box(SPRING_BLOCK, wb, t,
                                align=(Align.CENTER, Align.CENTER, Align.MIN))
    part += Pos(xb, 0, 0) * Box(SPRING_BLOCK, wb, t,
                                align=(Align.CENTER, Align.CENTER, Align.MIN))
    # the blade overlaps the blocks by 1 mm each side so the union is solid
    part += Pos(0, 0, t / 2) * Box(SPRING_FLEX + 2.0, width_mm, blade_mm)
    for x in (-xb, xb):
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


def tail_bar(length_mm: float) -> Part:
    """PLA bar along +X with an M3 hole near the tip for the counterweight."""
    if length_mm < TAIL_W or length_mm > BED_MM:
        raise ValueError("tail bar must fit the print bed")
    part = Pos(length_mm / 2, 0, 0) * Box(
        length_mm, TAIL_W, TAIL_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part -= Pos(length_mm - TAIL_W / 2, 0, 0) * Cylinder(M3 / 2, TAIL_T * 3)
    return part


def skid() -> Part:
    """Flat PLA pad (print in PLA): a low-friction contact surface."""
    return Box(SKID_L, SKID_W, SKID_T,
               align=(Align.CENTER, Align.CENTER, Align.MIN))


def spike() -> Part:
    """Tapered TPU tip (print in TPU) with grip grooves."""
    part = Cone(SPIKE_R, SPIKE_TIP_R, SPIKE_L,
                align=(Align.CENTER, Align.CENTER, Align.MIN))
    for z in (8.0, 16.0, 24.0):
        r = SPIKE_R + (SPIKE_TIP_R - SPIKE_R) * z / SPIKE_L
        part -= Pos(0, 0, z) * (Cylinder(r + 0.8, 1.0) - Cylinder(r - 0.8, 1.0))
    return Rot(Y=90) * part                        # tip along +X, as the module


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


def caster_wheel(radius_mm: float) -> Part:
    """Free PLA disc on an M3 axle: O-ring groove on the rim, no servo horn."""
    if not 15 <= radius_mm <= 35:
        raise ValueError("caster radius must be 15..35 mm")
    part = Cylinder(radius_mm, CASTER_T)
    part -= Cylinder(radius_mm + 1, 4) - Cylinder(radius_mm - 3, 5)
    part -= Cylinder(M3 / 2, CASTER_T * 3)
    return part


def caster_fork() -> Part:
    """U-shaped PLA fork: two arms on the wheel sides, one M3 axle hole through
    both arms (a single bore along X) and two M3 holes in the base."""
    w = CASTER_GAP + 2 * CASTER_ARM
    part = Box(w, CASTER_L, FORK_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part -= Pos(0, 5, 0) * Box(CASTER_GAP, 34, 3 * FORK_T)
    part -= Pos(0, 10, FORK_T / 2) * Rot(Y=90) * Cylinder(M3 / 2, 2 * w)
    for x in (-5, 5):
        part -= Pos(x, -17, 0) * Cylinder(M3 / 2, 3 * FORK_T)
    return part


def _mount(m, width_mm: float):
    """(x, y, yaw in degrees) of a torso mount. A mount given as (x, y) is
    turned to face outward: 90 degrees on a side edge, otherwise 0."""
    if len(m) == 3:
        return float(m[0]), float(m[1]), float(m[2])
    x, y = m
    return float(x), float(y), 90.0 if abs(abs(y) - width_mm / 2) < 1e-3 else 0.0


def _footprint(yaw_deg: float):
    """Axis-aligned extents (x, y) of the servo bracket footprint turned by yaw."""
    ol = SERVO_L + 2 * SERVO_CLEARANCE + 2 * WALL
    ow = SERVO_W + 2 * SERVO_CLEARANCE + 2 * WALL
    a = math.radians(yaw_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return ol * c + ow * s, ol * s + ow * c


def torso_plate(polygon, mounts) -> Part:
    """Extruded contour polygon (mm); each used core slot in `mounts` gets a tab
    under its servo bracket and the common mounting pattern. A mount is
    (x, y, yaw_deg): the slot position in mm and the yaw of the child on the
    core (its attachment angle), which turns both the tab footprint and the
    pattern. A plain (x, y) is turned 90 degrees on a side edge and 0
    elsewhere. The bracket is centred on the slot as in the simulation, so part
    of it overhangs the edge and the tab carries it."""
    t = PLATE_T
    pts = [(float(x), float(y)) for x, y in polygon]
    part = extrude(Polygon(*pts, align=None), amount=t)
    x0, x1, y0, y1 = _plate_bounds(polygon, mounts)
    width_mm = max(float(y) for _, y in polygon) - min(float(y) for _, y in polygon)
    ol = SERVO_L + 2 * SERVO_CLEARANCE + 2 * WALL
    ow = SERVO_W + 2 * SERVO_CLEARANCE + 2 * WALL
    feet = []                                   # bracket footprints (x, y, extent x, extent y)
    for m in mounts:
        x, y, yaw = _mount(m, width_mm)
        fx, fy = _footprint(yaw)
        feet.append((x, y, fx, fy))
        part += Pos(x, y, 0) * Rot(Z=yaw) * Box(
            ol, ow, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for m in mounts:
        x, y, yaw = _mount(m, width_mm)
        part -= Pos(x, y, 0) * _mount_holes(t, yaw)
    # ESP32 board hole pattern, centred behind the origin.
    # PLACEHOLDER: 58 x 23 mm is a guess. Check it against your board
    # (candidates in docs/hardware-dados.md) before printing.
    for sx in (-29, 29):
        for sy in (-11.5, 11.5):
            part -= Pos(sx, sy, 0) * Cylinder(1.6, t * 3)
    # battery strap slot across the plate, as far back as the brackets allow
    strap_w = min(width_mm - 20, 30)
    def free(sx):
        return all(abs(sx - x) > fx / 2 + 4 or abs(y) > fy / 2 + strap_w / 2 + 2
                   for x, y, fx, fy in feet)
    length_mm = x1 - x0
    xs = [x0 + 12 + i for i in range(int(length_mm / 2 - 12 - 35))]
    strap = next((sx for sx in xs if free(sx)), None)
    if strap is not None:
        part -= Pos(strap, 0, 0) * Box(4, strap_w, t * 3)
    return part


def _plate_bounds(polygon, mounts):
    """Bounding box of torso_plate in mm, brackets included."""
    x0 = min(float(x) for x, _ in polygon)
    x1 = max(float(x) for x, _ in polygon)
    y0 = min(float(y) for _, y in polygon)
    y1 = max(float(y) for _, y in polygon)
    width_mm = y1 - y0
    for m in mounts:
        x, y, yaw = _mount(m, width_mm)
        fx, fy = _footprint(yaw)
        x0, x1 = min(x0, x - fx / 2), max(x1, x + fx / 2)
        y0, y1 = min(y0, y - fy / 2), max(y1, y + fy / 2)
    return x0, x1, y0, y1


def tile_plan(polygon, mounts):
    """Fewest equal whole-millimetre tiles of the torso that fit BED_MM:
    (x0, y0, tile width, tile height, nx, ny) in mm."""
    x0, x1, y0, y1 = _plate_bounds(polygon, mounts)
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


def torso_tiles(polygon, mounts) -> list[Part]:
    """Printable torso pieces: torso_plate when its bounding box fits BED_MM,
    else the fewest equal tiles (grid along X and Y), each cut drilled for a
    splice."""
    plate = torso_plate(polygon, mounts)
    x0, y0, tw, th, nx, ny = tile_plan(polygon, mounts)
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

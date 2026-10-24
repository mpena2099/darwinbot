"""
parts.py
--------
Parametric printable parts (build123d) for the real DarwinBot: servo bracket,
link, TPU foot, wheel and torso plate. All dimensions in mm.

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
PLATE_T = 4.0           # link / plate thickness
LINK_W = 20.0           # link width = end boss diameter
M3 = 3.2                # M3 clearance hole
LINK_LENGTHS = (60, 80, 100, 120)
WHEEL_T = 10.0          # wheel thickness
FOOT_R = 9.0            # foot cap radius
FOOT_SOCKET_D = 12.0    # PLACEHOLDER: shin tube/socket diameter


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


def link(length_mm: float) -> Part:
    """Plate between two servo axes (horn at x=0, next bracket at x=length)."""
    if length_mm < LINK_W:
        raise ValueError("link too short")
    body = Pos(length_mm / 2, 0, 0) * (
       
        Box(length_mm + LINK_W, LINK_W, PLATE_T, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    part = fillet(body.edges().filter_by(lambda e: abs(e.length - PLATE_T) < 1e-6), LINK_W / 2 - 0.01)
    part -= _horn_holes(PLATE_T)
    # bracket end: pair of M3 holes plus a central slot for the cable
    for dy in (-6, 6):
        part -= Pos(length_mm, dy, 0) * Cylinder(M3 / 2, PLATE_T * 3)
    part -= Pos(length_mm, 0, 0) * Cylinder(HORN_BORE_D / 2, PLATE_T * 3)
    return part


def foot() -> Part:
    """Rounded cap (print in TPU): hemisphere over a socket for the shin."""
    with BuildPart() as b:
        Sphere(FOOT_R)
    dome = b.part - Pos(0, 0, -FOOT_R) * Box(4 * FOOT_R, 4 * FOOT_R, 2 * FOOT_R)
    stem = Pos(0, 0, 0) * Cylinder(FOOT_R, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = dome + stem
    part -= Pos(0, 0, 3) * Cylinder(FOOT_SOCKET_D / 2, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))
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


def torso_plate(length_mm: float, width_mm: float, hip_positions) -> Part:
    """Base plate; each hip gets a bracket mount (4 M3 holes) around its position."""
    t = PLATE_T
    part = Box(length_mm, width_mm, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = fillet(part.edges().filter_by(lambda e: abs(e.length - t) < 1e-6), 5)
    ol = SERVO_L + 2 * SERVO_CLEARANCE + 2 * WALL
    ow = SERVO_W + 2 * SERVO_CLEARANCE + 2 * WALL
    for x, y in hip_positions:
        for sx in (-1, 1):
            for sy in (-1, 1):
                part -= Pos(x + sx * (ol / 2 - 3), y + sy * (ow / 2 - 3), 0) * Cylinder(M3 / 2, t * 3)
    # ESP32 board: 58 x 23 mm hole pattern, centred behind the origin
    for sx in (-29, 29):
        for sy in (-11.5, 11.5):
            part -= Pos(sx, sy, 0) * Cylinder(1.6, t * 3)
    # battery strap slot across the plate, near the rear edge
    part -= Pos(-length_mm / 2 + 12, 0, 0) * Box(4, min(width_mm - 20, 30), t * 3)
    return part


def export_stl(part: Part, path) -> None:
    """Write part as a binary STL."""
    _export_stl(part, str(path))

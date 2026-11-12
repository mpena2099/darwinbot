"""
parts.py
--------
Parametric printable parts (build123d) for the real DarwinBot: servo cradle,
link with its U fork, elbow plate, wedge, TPU foot, wheel and torso plate (the
extruded contour polygon, one mount per used core slot). All dimensions in mm.

Conventions: a link lies along +X with the horn axis at the origin and the
next servo axis at x = length_mm, or at the end of its L/arc path; servo and
wheel axes point along Z. A cradle belongs to the parent of its servo and a U
link to the child, as in the simulation. Feet come in the FOOT_SHAPES of morphology.py.
Minimum wall 2 mm; slots and holes are vertical, so little needs support.
"""

import math

import numpy as np
from build123d import (Align, Box, BuildPart, Circle, Cone, Cylinder, Location,
                       Locations, gp_Trsf,
                       Part, Polygon, Pos, Rot, Sphere, SlotOverall, extrude,
                       export_stl as _export_stl)

from darwinbot import dims, hardware

# --- Feetech STS3215, from the SO-ARM100 STEP model (STS3215_03a, 2025) and the
# Waveshare ST3215 STEP and drawing (2022), which agree on these values ------
# (the SO-ARM100 model is Apache-2.0), not yet checked on a real servo: confirm
# every value below with calipers before printing. The body frame has x along
# the body length and z along the output axis, the origin at the body centre.
SERVO_L = hardware.SERVO.length          # body length (x), towards the output axis
SERVO_W = hardware.SERVO.width          # body width (y)
SERVO_H = hardware.SERVO.height          # envelope along z, horns included
SERVO_BODY_H = hardware.SERVO.body_height     # body alone, from z = -15.5 to z = +15.2
SERVO_Z0 = hardware.SERVO.z0        # face on the mounting side
SERVO_AXIS_X = hardware.SERVO.axis_x     # output axis offset from the body centre, both horns
SERVO_CLEARANCE = hardware.SERVO.clearance   # printing slack per side
HORN_BOLT_CIRCLE = hardware.SERVO.horn_bolt_circle # 4 holes on a 9.9 mm square
HORN_HOLE_YAW = hardware.SERVO.horn_hole_yaw    # hole pattern turned to the body axes
HORN_N_HOLES = hardware.SERVO.horn_n_holes
HORN_HOLE_D = hardware.SERVO.horn_hole_d       # M2 clearance
HORN_BORE_D = hardware.SERVO.horn_bore_d       # clearance for the centre screw head
SERVO_M2 = hardware.SERVO.m2          # M2 clearance for the body mounting holes
SERVO_TAP_X = hardware.SERVO.tap_x   # tapped M2 holes on the mounting face; the far
SERVO_TAP_Y = hardware.SERVO.tap_y           # face has them at x = +5.0 and -15.7
# ---------------------------------------------------------------------------

WALL = dims.WALL_MM     # wall thickness (>= 2)
PLATE_T = dims.PLATE_T_MM   # torso plate thickness (its tabs carry the edge servos)
LINK_T = 6.0            # default link / elbow plate thickness: 4 mm bends at ~36 MPa under the stall torque, 6 mm at ~16
LINK_W = 20.0           # default link width = end boss diameter
M3 = dims.M3_MM          # M3 clearance hole
BED_MM = float(round(hardware.PRINTER.bed * 1000.0, 6))          # largest part that fits a common print bed
SPLICE_W = 24.0         # splice strip width, across the cut
SPLICE_EDGE = 12.0      # splice hole distance from the strip ends
SPLICE_HOLE = 8.0       # splice hole distance from the cut on each side
WHEEL_T = dims.WHEEL_T_MM   # least wheel width: holds the horn screws and the O-ring groove with 3 mm walls
WHEEL_MAX_T = dims.WHEEL_MAX_T_MM   # widest wheel
CASTER_T = dims.CASTER_T_MM   # caster wheel thickness
CASTER_GAP = CASTER_T + 1.2   # clearance between the fork arms
CASTER_ARM = 7.5        # fork arm width, keeps >= 2 mm around the axle hole
FORK_T = 10.0          # fork thickness: keeps the axle bore 3 mm clear of the faces
CASTER_L = 44.0         # fork length along the arms
FOOT_R = 9.0            # default foot cap radius
FOOT_SOCKET_D = 12.0    # PLACEHOLDER: shin tube/socket diameter
FOOT_MIN_R = 8.6        # keeps the wall around the socket >= 2 mm
SPRING_FLEX = dims.SPRING_FLEX_MM   # TPU blade length
MOUNT_SIDE = dims.MOUNT_SIDE_MM   # M3 mounting pattern: 4 holes on a square of this side
MOUNT_MIN_W = MOUNT_SIDE + M3 + 2 * WALL   # narrowest part that holds the pattern
# end boss of a link or elbow, wide enough for the horn holes and for the M3
# pattern of the bracket bolted on the far side
END_BOSS_D = dims.END_BOSS_D_MM
SPRING_BLOCK = MOUNT_MIN_W   # mounting block length at each end of the blade
TAIL_W = dims.TAIL_W_MM   # tail bar width
TAIL_T = dims.TAIL_T_MM    # tail bar thickness
SKID_L, SKID_W, SKID_T = dims.SKID_L_MM, dims.SKID_W_MM, dims.SKID_T_MM   # low-friction pad
SPIKE_L, SPIKE_R = dims.SPIKE_L_MM, dims.SPIKE_R_MM               # TPU tip, tapered to SPIKE_TIP_R
SPIKE_TIP_R = 1.5
# --- cradle and U link (see docs/exporting.md) ------------------------------
HORN_T = dims.HORN_T_MM            # horn thickness, from the simulation's visual dress
CRADLE_GAP = dims.CRADLE_GAP_MM   # mm of air between a printed part and the servo body
CRADLE_T = dims.CRADLE_T_MM   # cradle wall and cap thickness
HORN_OPEN_R = 8.5       # horn opening in a cradle cap
ARM_W = dims.ARM_HALF_W_MM   # half width of a U link arm, across the link plane
ARM_T = dims.ARM_T_MM      # U link arm thickness, on each horn face
ARM_GAP = dims.ARM_GAP_MM    # air between a cradle cap and a U link arm
SLEEVE_R = 19.2         # the cradle sleeve and caps are trimmed to this radius about
                        # the output axis in front of it: the servo corner is at 16.1
                        # and the sleeve keeps its 2.5 mm wall there
FOOT_SWEEP_R = 23.6     # the cradle foot keeps inside this radius, in front of the axis
BRIDGE_X = dims.BRIDGE_X_MM   # where the U link crosses the servo front, along the link: it
                        # clears the foot, so the link turns the whole joint range
BRIDGE_T = dims.BRIDGE_T_MM   # U link bridge thickness along the link
MOUNT_DROP = dims.MOUNT_DROP_MM   # joint axis to the torso plate face: clears a U link arm (21.55)
MOUNT_FOOT_T = dims.MOUNT_FOOT_T_MM   # cradle foot thickness
WEDGE_T = dims.WEDGE_T_MM    # thinnest edge of a rolled-mount wedge
BACK_FOOT_X = dims.BACK_FOOT_X_MM   # a cradle whose output axis stands vertical bolts down behind
                        # the servo: its M3 pattern is this far behind the axis, out of
                        # the circle the U link arms turn in
# how far a cradle reaches back from the child servo axis: its back wall is
# where a link plate or an elbow deck stops
CRADLE_BACK = dims.CRADLE_BACK_MM
TONGUE_T = dims.TONGUE_T_MM   # elbow tongue lapped on the link tip (<= 0.5 mm)
TONGUE_Z = dims.TONGUE_HALF_H_MM   # tongue half height, clear of the link tip holes
STRUT_HW = dims.STRUT_HALF_W_MM   # elbow strut half width along the link
SPIGOT_R = 5.5          # link tip spigot inside a TPU foot socket
SPIGOT_L = 14.0


def horn_hole_centres() -> list:
    """(x, y) of the HORN_N_HOLES horn screw holes about the output axis: at
    HORN_HOLE_YAW to the body axes, so a 9.9 mm square on the 14 mm circle."""
    r = HORN_BOLT_CIRCLE / 2
    return [(r * math.cos(math.radians(HORN_HOLE_YAW + 360.0 * i / HORN_N_HOLES)),
             r * math.sin(math.radians(HORN_HOLE_YAW + 360.0 * i / HORN_N_HOLES)))
            for i in range(HORN_N_HOLES)]


def _horn_holes(thickness: float) -> Part:
    """Cutter: horn bore plus the screw circle, centred on the origin."""
    with BuildPart() as c:
        Cylinder(HORN_BORE_D / 2, thickness * 3)
        for x, y in horn_hole_centres():
            with Locations((x, y)):
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


def servo_body(axis: str = "pitch"):
    """(centre, half extents) of the printed servo body in the module frame,
    for a joint turning about `axis` (the simulation's morphology.servo_box
    with the printed body height)."""
    c = (-SERVO_AXIS_X, 0.0, 0.0)
    hb = SERVO_BODY_H / 2
    if axis == "pitch":
        return c, (SERVO_L / 2, SERVO_W / 2, hb)
    if axis == "yaw":
        return c, (SERVO_L / 2, hb, SERVO_W / 2)
    return (0.0, 0.0, -SERVO_AXIS_X), (hb, SERVO_W / 2, SERVO_L / 2)


def _axis_rot(axis: str):
    """Rotation from the canonical frame (output axis along +Z, body centre at
    -SERVO_AXIS_X along X) to the module frame of a joint about `axis`."""
    if axis == "pitch":
        return Rot(Z=0)
    if axis == "yaw":
        return Rot(X=-90)
    return Rot(X=180) * Rot(Y=90)


def _boxc(x0, x1, y0, y1, z0, z1) -> Part:
    """Box between two opposite corners, along the axes."""
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(
        abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))


def _outside_circle(radius: float) -> Part:
    """Cutter in the cradle frame: everything beyond `radius` of the output axis
    (the canonical Z axis) from just behind the axis forwards, where the link
    that turns about the servo sweeps."""
    return _boxc(-3.0, 300.0, -300.0, 300.0, -300.0, 300.0) \
        - Cylinder(radius, 600.0)


def _cradle_shell(axis: str, mount: bool, side: int = 1,
                  drop: float = None, shift: float = 0.0, back: bool = False):
    """Cradle around a servo in the module frame, before its bore is cut:
    (solid, bore). The sleeve walls wrap the four faces across the output axis
    and leave both horn faces open; a cap on each horn face carries the M2
    screws into the body's tapped holes. With `mount`, a foot on the module
    `side` * Y reaches the torso plate level with the M3 mounting pattern on
    the output axis. The body box and the caps do not change with `side`, so
    the two are the same part turned about the module X axis."""
    cx, cy, cz = -SERVO_AXIS_X, 0.0, 0.0          # canonical: output axis on Z
    hx, hy, hz = SERVO_L / 2, SERVO_W / 2, SERVO_BODY_H / 2
    hx, hy, hz = hx + SERVO_CLEARANCE, hy + SERVO_CLEARANCE, hz + SERVO_CLEARANCE
    part = _boxc(cx - hx - CRADLE_T, cx + hx + CRADLE_T,
                 cy - hy - CRADLE_T, cy + hy + CRADLE_T,
                 cz - hz - CRADLE_T, cz + hz + CRADLE_T)
    bore = _boxc(cx - hx, cx + hx, cy - hy, cy + hy,
                 cz - 4 * hz, cz + 4 * hz)
    part -= bore
    for s in (-1, 1):
        z = cz + s * (hz + CRADLE_T / 2)
        cap = _boxc(cx - hx - CRADLE_T, cx + hx + CRADLE_T,
                    cy - hy - CRADLE_T, cy + hy + CRADLE_T,
                    z - CRADLE_T / 2, z + CRADLE_T / 2)
        cap -= Pos(cx, cy, z) * Cylinder(HORN_OPEN_R, 4 * CRADLE_T)
        for tx in SERVO_TAP_X:
            for ty in (-SERVO_TAP_Y, SERVO_TAP_Y):
                cap -= Pos(cx + tx, ty, z) * Cylinder(SERVO_M2 / 2, 4 * CRADLE_T)
        part += cap
    # in front of the axis a link plate and its U fork turn about it: keep the
    # sleeve inside the circle they clear
    part -= _outside_circle(SLEEVE_R)
    if mount and back:
        # foot under the lower cap, behind the servo, with the M3 pattern
        # BACK_FOOT_X behind the output axis
        zt = cz - hz - CRADLE_T
        wb = MOUNT_SIDE + M3 + 3.0
        foot = _boxc(-BACK_FOOT_X - wb / 2, -BACK_FOOT_X + wb / 2,
                     -(hy + CRADLE_T), hy + CRADLE_T, -(MOUNT_DROP if drop is None else drop), zt + 0.5)
        part += foot
        part -= Pos(-BACK_FOOT_X, 0, 0) * _mount_holes(4 * MOUNT_DROP)
    elif mount:
        # foot under the module side * Y, plate face `drop` from the output axis
        y0 = cy - hy - CRADLE_T
        y1 = -(MOUNT_DROP if drop is None else drop)
        wf, wd = MOUNT_SIDE + 2 * WALL + 12.0, MOUNT_SIDE + M3 + 3.0
        # `shift` slides the foot along the output axis, so the M3 pattern of a
        # tilted foot hangs plumb under the axis, over the wedge; the foot stays
        # inside the caps, clear of the U link arms
        foot = _boxc(-wf / 2, wf / 2, y1, y1 + MOUNT_FOOT_T,
                     shift - wd / 2, shift + wd / 2)
        post = _boxc(-wf / 2, wf / 2, y0, y1 + MOUNT_FOOT_T,
                     min(shift, 0.0) - wd / 2, max(shift, 0.0) + wd / 2)
        holes = Pos(0, y1 + MOUNT_FOOT_T / 2, shift) * Rot(X=90) * \
            _mount_holes(MOUNT_FOOT_T)
        if side < 0:
            foot, post, holes = (Rot(X=180) * foot, Rot(X=180) * post,
                                 Rot(X=180) * holes)
        keep = (post + foot) - _outside_circle(FOOT_SWEEP_R)
        part += keep
        part -= holes
    return _axis_rot(axis) * part, _axis_rot(axis) * bore


def cradle(axis: str = "pitch", mount: bool = True, side: int = 1,
           drop: float = None, shift: float = 0.0, back: bool = False) -> Part:
    """Printed cradle holding one STS3215, in the module frame of a joint about
    `axis`: a sleeve around the body with a cap on each horn face, the four M2
    holes into the tapped face, and (with `mount`) a foot on the module
    `side` * Y down to the torso plate with the common M3 pattern on the
    output axis, its face `drop` mm from the axis (default: MOUNT_DROP, the
    torso plate level; shorter leaves room for a wedge on a rolled mount, and
    `shift` slides the foot along the output axis to keep its pattern plumb). It
    belongs to the parent of the servo: bolted to a torso tab, printed as one
    piece with a link tip, or at the end of an elbow plate."""
    shell, bore = _cradle_shell(axis, mount, side, drop, shift, back)
    return shell - bore


def _axis_roll(spec):
    """(axis, roll in degrees) of a cradle given as an axis name or as the
    pair (axis, roll about the module X)."""
    return (spec, 0.0) if isinstance(spec, str) else (spec[0], float(spec[1]))


def _u_root(axis: str, width_mm: float, thick_mm: float):
    """U link fork at the link root, plus the bridge into the link plate.
    Returns (solid, x of the plate start). The arms lie on the horn faces,
    outside the cradle caps, and carry the horn hole pattern on the output
    axis. A link about X (roll) has one arm, on the main horn face only."""
    d = SERVO_BODY_H / 2 + HORN_T + ARM_GAP
    hw = max(ARM_W, width_mm / 2 + 1.0)
    front = BRIDGE_X + BRIDGE_T
    part = None

    def add(b):
        nonlocal part
        part = b if part is None else part + b

    def horn(s):
        hole = _horn_holes(ARM_T)
        if axis == "pitch":
            return Pos(0, 0, s * (d + ARM_T / 2)) * hole
        if axis == "yaw":
            return Pos(0, s * (d + ARM_T / 2), 0) * hole
        return Pos(s * (d + ARM_T / 2), 0, 0) * Rot(Y=90) * hole

    if axis == "pitch":
        for s in (-1, 1):
            add(_boxc(-hw, front, -hw, hw, s * d, s * (d + ARM_T)))
            part -= horn(s)
        add(_boxc(BRIDGE_X, front, -thick_mm / 2, thick_mm / 2,
                  -(d + ARM_T), d + ARM_T))
    elif axis == "yaw":
        for s in (-1, 1):
            add(_boxc(-hw, front, s * d, s * (d + ARM_T), -hw, hw))
            part -= horn(s)
        add(_boxc(BRIDGE_X, front, -(d + ARM_T), d + ARM_T,
                  -thick_mm / 2, thick_mm / 2))
    else:
        # about X: one arm, on the main horn face; the far end of the servo is
        # where the parent holds it, so the link carries no strap there. The
        # plate runs on from this arm, along the output axis
        add(_boxc(d, d + ARM_T, -hw, hw, -hw, BRIDGE_X + ARM_T))
        part -= horn(1)
        front = d + ARM_T / 2
    return part, front


def link(length_mm: float, width_mm: float = LINK_W,
         thick_mm: float = LINK_T, profile: str = "straight",
         bend_deg: float = 0.0, axis: str = None, foot_r: float = None,
         tip_axis: str = None, elbow: tuple = ()) -> Part:
    """Plate between two servo axes (horn at x=0, next servo at the tip).
    With `axis` the root is a U link that clasps both servo horns (a joint at
    this link's root); otherwise it is a bolted plate end. `foot_r` frees the
    tip for a TPU foot: the plate ends in a spigot inside the foot socket.
    `tip_axis` prints the cradle of a child servo at the tip, as one piece.
    `elbow` is (right, left, bars, rigid): the fin edge on each side of the tip (or
    None) where the tongue of a side servo's elbow plate is bolted, the tip
    narrowing to a fin drilled for the M3 bolts; and the passive side modules
    printed as one piece with the link, each (signed distance, angle, foot
    radius, foot shape): a bar from the tip to the module, ending in a spigot
    for a foot; and the plain links hung on this one, each (position mm,
    rotation as nine numbers, link arguments), printed as one piece with it.
    An L link is two equal bars and an arc three, turned about Z so the whole
    path keeps its length."""
    if length_mm < width_mm:
        raise ValueError("link too short")
    pts = _link_path_mm(length_mm, profile, bend_deg)
    part = None
    for a, b in zip(pts, pts[1:]):
        yaw = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        bar = Pos(a[0], a[1], 0) * Rot(Z=yaw) * _bar(0.0, seg, width_mm, thick_mm)
        part = bar if part is None else part + bar
    part = Pos(0, 0, -thick_mm / 2) * part            # plate centred on z = 0
    part -= Pos(0, 0, -thick_mm / 2) * _horn_holes(thick_mm)
    tip = pts[-1]
    tip_deg = 0.0 if profile == "straight" else bend_deg
    if elbow[:2] != () and any(e is not None for e in elbow[:2]):
        frame = Pos(tip[0], tip[1], 0) * Rot(Z=tip_deg)
        # the cut spans the child's cradle along the link, and at least the
        # tip boss so no sliver of it is left
        reach = _boss_d(width_mm) / 2 + 1.0
        for spec in elbow[:2]:
            if spec is None:
                continue
            at, x0, x1 = spec
            x0, x1 = min(x0, -reach), max(x1, reach)
            part -= frame * (_boxc(x0, x1, at, 400, -400, 400)
                             if at > 0 else
                             _boxc(x0, x1, -400, at, -400, 400))
        for x in (-MOUNT_SIDE / 2, MOUNT_SIDE / 2):
            part -= frame * Pos(x, 0, 0) * Rot(X=90) \
                * Cylinder(M3 / 2, 80)
    else:
        part -= Pos(tip[0], tip[1], -thick_mm / 2) * Rot(Z=tip_deg) \
            * _link_end_pattern(thick_mm)
    if axis:
        root, front = _u_root(axis, width_mm, thick_mm)
        part -= _boxc(-400, front, -400, 400, -400, 400)
        part += root
    frame = Pos(tip[0], tip[1], 0) * Rot(Z=tip_deg)
    bars = _side_bars(frame, width_mm, thick_mm, elbow[2]) if len(elbow) > 2 else None
    if tip_axis:
        tip_axis, roll = _axis_roll(tip_axis)
        shell, bore = _cradle_shell(tip_axis, False)
        shell, bore = Rot(X=roll) * shell, Rot(X=roll) * bore
        # the plate stops at the cradle's back wall, so the child servo body
        # never shares space with it
        part -= frame * _boxc(-CRADLE_BACK, 400, -400, 400, -400, 400)
        part += frame * shell
        if bars is not None:
            part += bars
        part -= frame * bore
    elif bars is not None:
        part += bars
    for pos, rot, args in (elbow[3] if len(elbow) > 3 else ()):
        # a plain link hung on this one is rigid with it and overlaps its end:
        # printed as one piece, placed where the module tree puts it
        # the nine numbers are rounded to 4 digits: made orthonormal again, as
        # OpenCascade takes a slightly scaled turn for a scaled shape and its
        # fuse then silently drops the child link
        u, _, vt = np.linalg.svd(np.reshape(rot, (3, 3)))
        r = u @ vt
        trsf = gp_Trsf()
        trsf.SetValues(*r[0], pos[0], *r[1], pos[1], *r[2], pos[2])
        child = link(*args[:5], axis=args[5], foot_r=args[6], tip_axis=args[7],
                     elbow=args[8])
        part += child.moved(Location(trsf))
    if foot_r:
        # the foot slides over a spigot; clear its body out of the tip
        r, shape, *roll = foot_r if isinstance(foot_r, tuple) else (foot_r, "sphere")
        cut, spigot = _spigot_end(frame * Rot(X=roll[0] if roll else 0.0), r, shape)
        part -= cut
        part += spigot
    return part


def _spigot_end(frame, foot_r: float, shape: str = "sphere"):
    """(cutter, spigot) for a TPU foot at `frame`: the plate is cleared of the
    foot's body (it reaches back along -X: stem and hub) and a spigot runs from
    the plate into the socket, stopping 0.8 mm short of its floor, which
    absorbs the half millimetre the whole-mm lengths can move it."""
    r = max(foot_r, FOOT_MIN_R)
    body = foot(foot_r, shape)
    if shape == "sphere":
        body = Rot(Y=-90) * body
    box = body.bounding_box()
    back = box.min.X
    floor = {"sphere": -3.0, "blade": -r - 3.0, "pad": -r / 6 - 3.0,
             "claw": -3.0}[shape]
    cut = frame * body
    end = floor - 0.8
    top, bottom = end, back - 1.5          # spigot from `end` back into the plate
    spigot = frame * Pos((top + bottom) / 2, 0, 0) * Rot(Y=90) \
        * Cylinder(SPIGOT_R, top - bottom)
    return cut, spigot


def _side_bars(frame, width_mm: float, thick_mm: float, bars) -> Part:
    """Bars from a link tip (`frame`) to its passive side modules, printed as
    one piece with the link. Each bar is (signed distance, angle, foot radius,
    foot shape); the module sits `distance` to the side, turned by the angle
    (mirrored on the other side), and a foot gets a spigot."""
    out = None
    for dist, angle, foot_r, shape, *roll in bars:
        s = 1.0 if dist > 0 else -1.0
        bar = Pos(0, 0, -thick_mm / 2) * (Rot(Z=90.0 * s) * _bar(0.0, abs(dist), width_mm, thick_mm))
        # the bar's start boss belongs to its own side of the tip: the other
        # side may carry a fin and an elbow plate
        bar -= _boxc(-400, 400, -400, 0, -400, 400) if s > 0 else \
            _boxc(-400, 400, 0, 400, -400, 400)
        cut = spigot = None
        if foot_r:
            cut, spigot = _spigot_end(Pos(0, dist, 0) * Rot(Z=s * angle)
                                      * Rot(X=roll[0] if roll else 0.0), foot_r, shape)
            bar -= cut
            bar += spigot
        out = frame * bar if out is None else out + frame * bar
    return out


def _link_path_mm(length_mm: float, profile: str, bend_deg: float):
    """Polyline of a link in mm, starting at the origin, same as
    morphology.link_path."""
    pieces = {"straight": 1, "L": 2, "arc": 3}[profile]
    step = length_mm / pieces
    turn = math.radians(bend_deg) / (pieces - 1) if pieces > 1 else 0.0
    pts = [(0.0, 0.0)]
    angle = 0.0
    for k in range(pieces):
        if k:
            angle += turn
        x, y = pts[-1]
        pts.append((x + step * math.cos(angle), y + step * math.sin(angle)))
    return pts


def _link_end_pattern(thick_mm: float, yaw_deg: float = 0.0) -> Part:
    """Cutter at the origin: cable bore plus the mounting pattern."""
    return Cylinder(HORN_BORE_D / 2, thick_mm * 3) + _mount_holes(thick_mm, yaw_deg)


def _link_end_holes(x: float, y: float, angle_deg: float = 0.0,
                    thick_mm: float = PLATE_T) -> Part:
    """The bracket end pattern placed at (x, y) and turned about Z."""
    return Pos(x, y, 0) * _link_end_pattern(thick_mm, angle_deg)


def elbow_plate(distance_mm: float, angle_deg: float = 90.0,
                width_mm: float = LINK_W, thick_mm: float = LINK_T,
                axis: str = None, trim: float = None) -> Part:
    """Bar bolted on a link end (origin) that carries a side module
    `distance_mm` away along +Y. The near end is a thin tongue lapped on the
    link tip; the far end repeats the link end pattern, turned about Z by
    angle_deg - 90. With `axis` the far end is the cradle of the child servo,
    printed as one piece, and the near end is a vertical tongue bolted to the
    side of the link tip fin (two M3 bolts along Y shared by the elbows on both
    sides): the child servo body sits in the link plane, so the elbow keeps no
    material there and joins the cradle's back wall with a strut. `trim` is
    where the link tip fin ends on this side (default: the cradle back wall
    for a 90 degree angle)."""
    if distance_mm < width_mm:
        raise ValueError("elbow too short")
    if axis:
        axis, roll = _axis_roll(axis)
        frame = Pos(0, distance_mm, 0) * Rot(Z=angle_deg) * Rot(X=roll)
        at = (distance_mm - CRADLE_BACK - CRADLE_GAP) if trim is None else trim
        hx = SERVO_W / 2 + SERVO_CLEARANCE + CRADLE_T
        shell, bore = _cradle_shell(axis, False)
        box = (frame * shell).bounding_box()
        hz = max(1.0, min(-box.min.Z, box.max.Z) - 0.5)
        tongue = _boxc(-hx, hx, at - TONGUE_T, at + 0.1, -TONGUE_Z, TONGUE_Z)
        for x in (-MOUNT_SIDE / 2, MOUNT_SIDE / 2):
            tongue -= Pos(x, 0, 0) * Rot(X=90) * Cylinder(M3 / 2, 4 * TONGUE_Z)
        # a strut from the tongue to the cradle, wherever its back wall is for
        # this distance and angle; the servo bore and the horn space stay free
        strut = _boxc(-STRUT_HW, STRUT_HW, at, distance_mm, -hz, hz)
        strut -= frame * bore
        strut -= frame * Cylinder(HORN_OPEN_R + 1.5, 4 * hz)
        # the child's U link arms lie outside the caps, along its output axis
        cap = SERVO_BODY_H / 2 + SERVO_CLEARANCE + CRADLE_T + ARM_GAP / 2
        for sign in (-1, 1):
            strut -= frame * _axis_rot(axis) * _boxc(
                -400, 400, -400, 400, sign * cap, sign * 400)
        part = tongue + strut + frame * shell
        part -= frame * bore
        solids = sorted(part.solids(), key=lambda s: -s.volume)
        return part if len(solids) == 1 else solids[0]
    part = Rot(Z=90) * _bar(0.0, distance_mm, width_mm, thick_mm)
    part = Pos(0, 0, -thick_mm / 2) * part
    part -= Pos(0, 0, -thick_mm / 2) * _link_end_holes(0, 0, 0.0, thick_mm)
    # only the upper half of the link tip, and a thin tongue up to the bar
    part -= _boxc(-400, 400, -400, 0, -400, 400)
    part -= _boxc(-400, 400, -400, MOUNT_SIDE, -400, thick_mm / 2 - TONGUE_T)
    part -= Pos(0, distance_mm, -thick_mm / 2) \
        * _link_end_holes(0, 0, angle_deg - 90.0, thick_mm)
    return part

def wedge(angle_deg: float) -> Part:
    """Printed wedge between a torso tab and a cradle foot, taking up a slot
    roll that is not a multiple of 90 degrees: the common M3 mounting pattern
    through both faces, the bottom flat on the tab and the top turned by
    `angle_deg` about Y (the out-of-plane roll axis), WEDGE_T thick at its
    thin edge."""
    if abs(angle_deg) > 80.0:
        raise ValueError("wedge angle must be within 80 degrees")
    w = MOUNT_SIDE + 2 * WALL + 2.0
    a = abs(math.tan(math.radians(angle_deg)))
    top = WEDGE_T + a * w
    part = _boxc(-w / 2, w / 2, -w / 2, w / 2, 0, top)
    # the top face passes through (-w/2, WEDGE_T) and rises with x
    part -= Pos(-w / 2, 0, WEDGE_T) * Rot(Y=-abs(angle_deg)) \
        * Pos(0, 0, 2 * w) * Box(4 * w, 4 * w, 4 * w)
    part -= _mount_holes(2 * top)
    return Rot(Z=180) * part if angle_deg < 0 else part


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


def _socketed(shape: Part, back_x: float) -> Part:
    """Add the shin socket to a foot shape: a hub on its back face (x = back_x,
    the side the limb comes from) drilled for the same socket as the sphere."""
    wall = FOOT_SOCKET_D / 2 + 2
    hub = Pos(back_x, 0, 0) * Rot(Y=-90) * Cylinder(
        wall, 12, align=(Align.CENTER, Align.CENTER, Align.MIN))
    hole = Pos(back_x - 12, 0, 0) * Rot(Y=90) * Cylinder(
        FOOT_SOCKET_D / 2, 9, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return shape + hub - hole


def foot(radius_mm: float = FOOT_R, shape: str = "sphere") -> Part:
    """TPU tip in one of the FOOT_SHAPES. The sphere is the rounded cap over a
    socket for the shin. The blade is a long thin box, the pad a disc with its
    axis along X (so its flat face looks down when the limb points down) and
    the claw three toes spread in the XY plane. Scaled from the 9 mm foot; the
    socket keeps its diameter and the radius is clamped so the wall around it
    stays at least 2 mm."""
    r = max(radius_mm, FOOT_MIN_R)
    if shape == "blade":
        return _socketed(Box(2 * r, r / 3, r), -r)
    if shape == "pad":
        return _socketed(Rot(Y=90) * Cylinder(r, r / 3), -r / 6)
    if shape == "claw":
        part = None
        for a in (-30.0, 0.0, 30.0):
            toe = Rot(Z=a) * Rot(Y=90) * Cylinder(
                r / 5, 1.5 * r, align=(Align.CENTER, Align.CENTER, Align.MIN))
            end = Pos(1.5 * r * math.cos(math.radians(a)),
                      1.5 * r * math.sin(math.radians(a)), 0) * Sphere(r / 5)
            toe += end
            part = toe if part is None else part + toe
        return _socketed(part, 0.0)
    with BuildPart() as b:
        Sphere(r)
    dome = b.part - Pos(0, 0, -r) * Box(4 * r, 4 * r, 2 * r)
    stem = Pos(0, 0, 0) * Cylinder(r, 10 * r / FOOT_R,
                                   align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = dome + stem
    part -= Pos(0, 0, 3) * Cylinder(FOOT_SOCKET_D / 2, max(20.0, 3 * r),
                                    align=(Align.CENTER, Align.CENTER, Align.MIN))
    return part


RISER_FLANGE_T = 3.0     # base flange of a riser post
RISER_FLANGE_D = 28.0
RISER_OD = 20.0
RISER_CAP_T = 4.0        # solid top under the spigot
RISER_SPIGOT_L = 8.2     # into the foot socket, 0.8 mm short of its floor


def riser(height_mm: float) -> Part:
    """Post from the torso plate to a module on a core face slot. Z up from
    the plate face: a flange screwed through the face mount (the common M3
    pattern), a hollow column, and a spigot that goes into the foot socket at
    the top, `height_mm` from the plate."""
    h = float(height_mm)
    if h < 1.0:
        raise ValueError("riser too short")
    part = Cylinder(RISER_FLANGE_D / 2, min(RISER_FLANGE_T, h),
                    align=(Align.CENTER, Align.CENTER, Align.MIN))
    part += Cylinder(RISER_OD / 2, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    if h - RISER_CAP_T > RISER_FLANGE_T + 2.0:     # a short post stays solid
        part -= Cylinder(RISER_OD / 2 - 3.0, h - RISER_CAP_T,
                         align=(Align.CENTER, Align.CENTER, Align.MIN))
    part += Pos(0, 0, h - min(1.0, h)) * Cylinder(
        SPIGOT_R, RISER_SPIGOT_L + min(1.0, h), align=(Align.CENTER, Align.CENTER, Align.MIN))
    part -= _mount_holes(RISER_FLANGE_T)
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


def wheel(radius_mm: float, width_mm: float = 20.0) -> Part:
    """Disc on a servo horn; a groove on the rim holds an O-ring tyre. The
    width runs along the axle (Z), 10 to 60 mm."""
    if not 30 <= radius_mm <= 100:
        raise ValueError("wheel radius must be 30..100 mm")
    if not WHEEL_T <= width_mm <= WHEEL_MAX_T:
        raise ValueError("wheel width must be 10..60 mm")
    disc = Cylinder(radius_mm, width_mm)
    # groove: 3 mm deep, 4 mm wide, rectangular, at mid-thickness
    ring = Cylinder(radius_mm + 1, 4) - Cylinder(radius_mm - 3, 4 + 1)
    part = disc - ring
    part -= _horn_holes(width_mm)
    # lightening holes keep walls >= 2 mm: only when the disc is big enough
    if radius_mm >= 40:
        r_mid = (radius_mm - 3 + HORN_BOLT_CIRCLE / 2 + HORN_HOLE_D) / 2
        d = min(10.0, (radius_mm - 3 - HORN_BOLT_CIRCLE / 2 - HORN_HOLE_D) - 2 * WALL)
        if d > 4:
            for i in range(6):
                a = 2 * math.pi * (i + 0.5) / 6
                part -= Pos(r_mid * math.cos(a), r_mid * math.sin(a), 0) * Cylinder(d / 2, width_mm * 3)
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
    turned to face outward: +-90 degrees on a side edge, 180 at the rear and
    0 at the front. A fourth
    element True marks a face mount (see _face)."""
    if len(m) >= 3:
        return float(m[0]), float(m[1]), float(m[2])
    x, y = m
    if abs(abs(y) - width_mm / 2) < 1e-3:
        return float(x), float(y), 90.0 if y > 0 else -90.0
    return float(x), float(y), 0.0 if x >= 0 else 180.0


def _face(m) -> bool:
    """True for a mount on the top or bottom face of the torso: drilled through
    the plate with the mounting pattern, no edge tab."""
    return len(m) > 3 and bool(m[3])


def _back(m) -> float:
    """How far behind the slot position the mounting holes of a mount are: a
    servo whose output axis stands vertical bolts down behind itself."""
    return float(m[4]) if len(m) > 4 else 0.0


def _footprint_centre(x: float, y: float, yaw_deg: float):
    """Centre of the bracket footprint of a mount at (x, y): the body sits
    SERVO_AXIS_X behind the output axis along the yawed -X."""
    a = math.radians(yaw_deg)
    return x - SERVO_AXIS_X * math.cos(a), y - SERVO_AXIS_X * math.sin(a)


def _footprint(yaw_deg: float):
    """Axis-aligned extents (x, y) of the servo bracket footprint turned by yaw."""
    ol = SERVO_L + 2 * SERVO_CLEARANCE + 2 * WALL
    ow = SERVO_W + 2 * SERVO_CLEARANCE + 2 * WALL
    a = math.radians(yaw_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return ol * c + ow * s, ol * s + ow * c


def torso_plate(polygon, mounts, electronics: bool = True) -> Part:
    """Extruded contour polygon (mm); each used core slot in `mounts` gets a tab
    under its servo bracket and the common mounting pattern. A mount is
    (x, y, yaw_deg): the slot position in mm and the yaw of the child on the
    core (its attachment angle), which turns both the tab footprint and the
    pattern; a fourth element True (a face mount) drills the pattern only,
    with no tab. A plain (x, y) is turned 90 degrees on a side edge and 0
    elsewhere. The output axis sits on the slot as in the simulation and the
    bracket behind it, so part of it may overhang the edge and the tab carries it. `electronics=False` skips
    the board holes and battery strap, for a segment plate."""
    t = PLATE_T
    pts = [(float(x), float(y)) for x, y in polygon]
    part = extrude(Polygon(*pts, align=None), amount=t)
    x0, x1, y0, y1 = _plate_bounds(polygon, mounts)
    width_mm = max(float(y) for _, y in polygon) - min(float(y) for _, y in polygon)
    ol = SERVO_L + 2 * SERVO_CLEARANCE + 2 * WALL
    ow = SERVO_W + 2 * SERVO_CLEARANCE + 2 * WALL
    feet = []                                   # bracket footprints (x, y, extent x, extent y)
    for m in mounts:
        if _face(m):
            continue
        x, y, yaw = _mount(m, width_mm)
        fx, fy = _footprint(yaw)
        cx, cy = _footprint_centre(x, y, yaw)
        feet.append((cx, cy, fx, fy))
        part += Pos(cx, cy, 0) * Rot(Z=yaw) * Box(
            ol, ow, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for m in mounts:
        x, y, yaw = _mount(m, width_mm)
        b = _back(m)
        a = math.radians(yaw)
        part -= Pos(x - b * math.cos(a), y - b * math.sin(a), 0) \
            * _mount_holes(t, yaw)
    if not electronics:
        return part
    # Board hole pattern from the electronics profile, centred behind the origin.
    for sx, sy in hardware.hole_points(hardware.ELECTRONICS.holes):
        part -= Pos(sx, sy, 0) * Cylinder(hardware.ELECTRONICS.hole_d / 2, t * 3)
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
        if _face(m):
            continue
        x, y, yaw = _mount(m, width_mm)
        fx, fy = _footprint(yaw)
        x, y = _footprint_centre(x, y, yaw)
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

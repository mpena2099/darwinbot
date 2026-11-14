"""
dims.py
-------
The printed-part dimensions, in one place. The simulation (morphology.py,
builder.py), the part generators (parts.py) and the export (export.py) read
them from here, so a printed part and the shape the simulator collides cannot
drift apart. The printed part is the truth.

Every value is defined in metres; the `*_MM` twin is the same value in
millimetres, rounded to 1e-6 mm so that it is the exact decimal the part
generator uses. This module needs neither build123d nor MuJoCo.
"""

import math

from darwinbot import hardware


def mm(metres: float) -> float:
    """Metres to millimetres, rounded to the nearest nanometre."""
    return round(metres * 1000.0, 6)


def metres(millimetres: float) -> float:
    """Millimetres to metres, rounded so that 2.9 mm is exactly 0.0029."""
    return round(millimetres / 1000.0, 12)


# --- walls, bolts and the end boss of a link or elbow -----------------------
WALL = 0.0025               # wall thickness
M3 = 0.0032                 # M3 clearance hole
MOUNT_SIDE = 0.016          # M3 mounting pattern: 4 holes on a square of this side
WALL_MM, M3_MM, MOUNT_SIDE_MM = mm(WALL), mm(M3), mm(MOUNT_SIDE)
# end boss of a link or elbow, wide enough for the horn holes and for the M3
# pattern of the bracket bolted on the far side
END_BOSS_D_MM = max(
    hardware.SERVO.horn_bolt_circle + 2 * (hardware.SERVO.horn_hole_d / 2 + WALL_MM),
    MOUNT_SIDE_MM * math.sqrt(2) + M3_MM + 2 * WALL_MM)
END_BOSS_D = metres(END_BOSS_D_MM)
ELBOW_BOSS_R = END_BOSS_D_MM / 2 / 1000.0   # round end of a printed elbow plate

# --- servo, cradle and U link -----------------------------------------------
CRADLE_T = 0.0025           # cradle wall and cap thickness
CRADLE_GAP = 0.0002         # air between a printed part and the servo body
PRINT_WALL = metres(hardware.SERVO.clearance + CRADLE_T * 1000.0)   # air and wall around a servo
HORN_T = metres(hardware.SERVO.horn_thickness)
ARM_T = 0.003               # U link arm, one on each horn face
ARM_GAP = 0.0002            # air between a cradle cap and an arm
ARM_HALF_W = 0.0134         # half width of an arm across the link plane
BRIDGE_X = 0.024            # where the U link crosses the servo front
BRIDGE_T = 0.003
STRUT_HALF_W = 0.004        # elbow strut half width along the link
TONGUE_HALF_H = 0.0045      # elbow tongue half height
TONGUE_T = 0.00045          # elbow tongue lapped on a link tip (the simulation
                            # collides a thicker one: morphology.TONGUE_T)
MOUNT_DROP = 0.022          # joint axis to the torso plate face
MOUNT_FOOT_T = 0.003        # cradle foot thickness
WEDGE_T = 0.002             # thinnest edge of a rolled-mount wedge
WEDGE_W = metres(mm(MOUNT_SIDE) + 2 * mm(WALL) + 2.0)   # wedge side
BACK_FOOT_X = 0.031         # M3 pattern of a back foot behind the axis
# how far a cradle reaches back from the child servo axis: its back wall is
# where a link plate or an elbow deck stops
CRADLE_BACK_MM = round(
    hardware.SERVO.axis_x + hardware.SERVO.length / 2
    + hardware.SERVO.clearance + mm(CRADLE_T), 6)
CRADLE_BACK = metres(CRADLE_BACK_MM)

# --- torso and passive parts ------------------------------------------------
PLATE_THICK = 0.006         # torso plate and splice strip
TAIL_BAR_SIZE = (0.015, 0.006)       # width and thickness of the tail bar
SKID_SIZE = (0.06, 0.03, 0.006)      # low-friction pad
SPIKE_LENGTH = 0.03
SPIKE_RADIUS = 0.005
CASTER_WHEEL_T = 0.008      # caster wheel thickness
SPRING_FLEX_LEN = 0.012     # free TPU blade length
WHEEL_WIDTH_RANGE = (0.010, 0.060)   # wheel width; 10 mm is the least that holds the
                                     # horn screws (14 mm circle) and the 4 mm O-ring
                                     # groove with 3 mm walls

# --- the same values in millimetres, for parts.py and export.py -------------
ARM_T_MM, ARM_GAP_MM, HORN_T_MM = mm(ARM_T), mm(ARM_GAP), mm(HORN_T)
ARM_HALF_W_MM = mm(ARM_HALF_W)
BRIDGE_X_MM, BRIDGE_T_MM = mm(BRIDGE_X), mm(BRIDGE_T)
STRUT_HALF_W_MM, TONGUE_HALF_H_MM = mm(STRUT_HALF_W), mm(TONGUE_HALF_H)
TONGUE_T_MM = mm(TONGUE_T)
CRADLE_T_MM, CRADLE_GAP_MM = mm(CRADLE_T), mm(CRADLE_GAP)
MOUNT_DROP_MM, MOUNT_FOOT_T_MM = mm(MOUNT_DROP), mm(MOUNT_FOOT_T)
WEDGE_T_MM, WEDGE_W_MM, BACK_FOOT_X_MM = mm(WEDGE_T), mm(WEDGE_W), mm(BACK_FOOT_X)
PLATE_T_MM = mm(PLATE_THICK)
TAIL_W_MM, TAIL_T_MM = (mm(v) for v in TAIL_BAR_SIZE)
SKID_L_MM, SKID_W_MM, SKID_T_MM = (mm(v) for v in SKID_SIZE)
SPIKE_L_MM, SPIKE_R_MM = mm(SPIKE_LENGTH), mm(SPIKE_RADIUS)
CASTER_T_MM, SPRING_FLEX_MM = mm(CASTER_WHEEL_T), mm(SPRING_FLEX_LEN)
WHEEL_T_MM, WHEEL_MAX_T_MM = (mm(v) for v in WHEEL_WIDTH_RANGE)

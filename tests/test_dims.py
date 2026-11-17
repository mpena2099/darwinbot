"""The printed-part dimensions exist in parts.py (mm), morphology.py (m) and
export.py (mm): the three copies must agree."""
import pytest

pytest.importorskip("build123d")

from darwinbot import export, hardware, parts
from darwinbot import morphology as mo

S = hardware.SERVO


def _pairs():
    mm = 1000.0
    return [
        ("END_BOSS_D", mo.ELBOW_BOSS_R * 2 * mm, parts.END_BOSS_D),
        ("SERVO_BODY_H", mo.PRINT_BODY_H * mm, parts.SERVO_BODY_H),
        ("PRINT_WALL", mo.PRINT_WALL * mm, parts.SERVO_CLEARANCE + parts.CRADLE_T),
        ("ARM_T", mo.FORK_ARM_T * mm, parts.ARM_T),
        ("ARM_GAP", mo.FORK_ARM_GAP * mm, parts.ARM_GAP),
        ("HORN_T", mo.FORK_HORN_T * mm, parts.HORN_T),
        ("ARM_W", mo.FORK_ARM_HALF_W * mm, parts.ARM_W),
        ("BRIDGE_X", mo.FORK_BRIDGE_X * mm, parts.BRIDGE_X),
        ("BRIDGE_T", mo.FORK_BRIDGE_T * mm, parts.BRIDGE_T),
        ("STRUT_HW", mo.STRUT_HALF_W * mm, parts.STRUT_HW),
        ("TONGUE_Z", mo.TONGUE_HALF_H * mm, parts.TONGUE_Z),
        ("PLATE_T", mo.PLATE_THICK * mm, parts.PLATE_T),
        ("PLATE_T export", export.PLATE_T, parts.PLATE_T),
        ("TAIL_W", mo.TAIL_BAR_SIZE[0] * mm, parts.TAIL_W),
        ("TAIL_T", mo.TAIL_BAR_SIZE[1] * mm, parts.TAIL_T),
        ("SKID_L", mo.SKID_SIZE[0] * mm, parts.SKID_L),
        ("SKID_W", mo.SKID_SIZE[1] * mm, parts.SKID_W),
        ("SKID_T", mo.SKID_SIZE[2] * mm, parts.SKID_T),
        ("SPIKE_L", mo.SPIKE_LENGTH * mm, parts.SPIKE_L),
        ("SPIKE_R", mo.SPIKE_RADIUS * mm, parts.SPIKE_R),
        ("CASTER_T", mo.CASTER_WHEEL_T * mm, parts.CASTER_T),
        ("SPRING_FLEX", mo.SPRING_FLEX_LEN * mm, parts.SPRING_FLEX),
        ("WHEEL_T", mo.WHEEL_WIDTH_RANGE[0] * mm, parts.WHEEL_T),
        ("WHEEL_MAX_T", mo.WHEEL_WIDTH_RANGE[1] * mm, parts.WHEEL_MAX_T),
        ("cradle length", mo.cradle_box("pitch")[0][0] * 2 * mm - 2 * mo.PRINT_WALL * mm, S.length),
        ("cradle width", mo.cradle_box("pitch")[0][1] * 2 * mm - 2 * mo.PRINT_WALL * mm, S.width),
        ("export SERVO_BODY", export.SERVO_BODY[0], parts.SERVO_L),
        ("export SERVO_BODY w", export.SERVO_BODY[1], parts.SERVO_W),
        ("export SERVO_BODY h", export.SERVO_BODY[2], parts.SERVO_BODY_H),
        ("export SERVO_AXIS_X", export.SERVO_AXIS_X, parts.SERVO_AXIS_X),
        ("CRADLE_SLACK", export.CRADLE_SLACK, parts.SERVO_CLEARANCE + parts.CRADLE_T),
        ("FLOOR_T", export.FLOOR_T, parts.MOUNT_FOOT_T),
        ("CRADLE_BACK", export.CRADLE_BACK, parts.CRADLE_BACK),
        ("CRADLE_GAP", export.CRADLE_GAP, parts.CRADLE_GAP),
        ("MOUNT_DROP", export.MOUNT_DROP, parts.MOUNT_DROP),
        ("WEDGE_T", export.WEDGE_T, parts.WEDGE_T),
        ("WEDGE_W", export.WEDGE_W, parts.MOUNT_SIDE + 2 * parts.WALL + 2),
        ("BACK_FOOT_X", export.BACK_FOOT_X, parts.BACK_FOOT_X),
        ("TONGUE_T", export.TONGUE_T, parts.TONGUE_T),
    ]


@pytest.mark.parametrize("name,a,b", [pytest.param(*p, id=p[0]) for p in _pairs()])
def test_copies_agree(name, a, b):
    assert a == pytest.approx(b, abs=1e-9), name

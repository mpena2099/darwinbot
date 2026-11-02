import numpy as np


def test_hard_plastic_slides_and_tpu_grips():
    """Servo housings, links and the torso get the plastic friction, which wins
    the contact over the ground; feet and wheels keep the TPU friction."""
    import mujoco
    from darwinbot import builder, morphology as mo
    xml, _ = builder.build_mjcf(mo.template(2))
    m = mujoco.MjModel.from_xml_string(xml)
    robot = m.geom_contype == 2
    plastic = robot & (m.geom_priority == 2)
    assert plastic.any()
    assert np.allclose(m.geom_friction[plastic, 0], builder.PLASTIC_FACTOR)
    assert (m.geom_friction[robot & (m.geom_priority == 0), 0] == 1.0).all()


def test_servo_reaches_datasheet_speed():
    from darwinbot.controller import SERVO_SPEED, SERVO_TORQUE
    from darwinbot.builder import JOINT_DAMPING, JOINT_FRICTION
    top = (SERVO_TORQUE - JOINT_FRICTION) / JOINT_DAMPING
    assert 0.9 * SERVO_SPEED < top <= SERVO_SPEED

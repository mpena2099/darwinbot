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


def test_fully_blocked_joint_keeps_a_limit(monkeypatch):
    """A joint that clashes on both sides at once still gets a tiny range: a 0 0
    range would read as "no limit" to MuJoCo."""
    import mujoco
    from darwinbot import builder, morphology as mo
    morph = mo.template(2)
    real = builder.AssemblyRules.clashes

    def clashes(self, model, data, fixed=False, first=False):
        return True if not fixed else real(self, model, data, fixed=fixed, first=first)

    monkeypatch.setattr(builder.AssemblyRules, "clashes", clashes)
    monkeypatch.setattr(builder, "_RANGES", {})
    ranges = builder.joint_ranges(morph)
    assert ranges and all(r == (-builder.BLOCKED_SPAN, builder.BLOCKED_SPAN)
                          for r in ranges.values())
    xml, layout = builder.build_mjcf(morph, ranges=ranges)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.jnt_limited[m.actuator_trnid[:, 0]].all()
    assert m.actuator_ctrllimited[:len(layout)].all()


def _body_with_tip_and_side_servo():
    """template(2) with a servo on the tip of the hip as well as the knee on
    its side: the knee's elbow fin is printed on the side of the tip cradle."""
    from darwinbot import morphology as mo
    morph = mo.template(2)
    morph.nodes[2].offset = 0.5           # the knee hangs 62 mm out on the side
    i = mo._free_id(morph)
    morph.nodes[i] = mo.Node(i, "joint", 1, 0, axis="pitch", length=0.08, flip=True)
    return morph


def _side_links(rules, parent):
    """Geoms of the printed link of `parent`: its plate, bosses, fork and the
    fins of its side mounts (role "link", printed with it)."""
    return [g for g in range(len(rules.role))
            if rules.role[g] == "link" and rules.part[g] == ("link", parent.id)]


def test_every_elbow_tongue_reaches_its_link():
    """A side servo hangs on an elbow whose tongue is bolted to a fin of the
    link tip: no strut or tongue may float, leaving the moulded gap a body
    without the fin had."""
    import mujoco
    from darwinbot import builder
    morph = _body_with_tip_and_side_servo()
    xml, _ = builder.build_mjcf(morph, ranges={})
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rules = builder.AssemblyRules(morph, model)
    fromto = np.zeros(6)
    tested = 0
    for parent in morph.nodes.values():
        if parent.kind not in ("joint", "link", "spring"):
            continue                      # a torso mount hangs on a tab, no fin
        side = [c for s, c in morph.child_map().get(parent.id, {}).items()
                if s and morph.nodes[c].kind in ("joint", "segment", "wheel")]
        if not side:
            continue
        link = _side_links(rules, parent)
        assert any(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                   .startswith("fin") for g in link), parent.id
        for g in range(model.ngeom):
            if rules.role[g] not in ("strut", "tongue"):
                continue
            if rules.parent[rules.node[g]] != parent.id:
                continue
            d = min(mujoco.mj_geomDistance(model, data, g, h, 1.0, fromto)
                    for h in link)
            assert d <= 0.0005, (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, g), d)
            tested += 1
    assert tested >= 2


def test_a_missing_fin_is_not_buildable(monkeypatch):
    """The fin is what the tongue bolts to: without it the elbow floats, and
    the spawn check refuses the body. The fin carries the link plate's mass."""
    import mujoco
    from darwinbot import builder, sim
    morph = _body_with_tip_and_side_servo()
    xml, _ = builder.build_mjcf(morph, ranges={})
    model = mujoco.MjModel.from_xml_string(xml)
    assert sim.is_buildable(model, morph)
    monkeypatch.setattr(builder, "_fin_geom", lambda *args, **kwargs: "")
    xml, _ = builder.build_mjcf(morph, ranges={})
    assert 'name="fin' not in xml
    bare = mujoco.MjModel.from_xml_string(xml)
    assert not sim.is_buildable(bare, morph)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "n1")
    assert model.body_mass[body] > bare.body_mass[body]


"""
scenarios/surfaces.py
---------------------
Phase 2 — three surface zones in sequence along the X axis.

Layout (top view, robot travels left → right):

  X=0..3  : smooth wood floor   (low friction)
  X=3..6  : rough concrete      (high friction)
  X=6..9  : carpet              (very high friction + damping)

Each zone is a flat box with different dynamics parameters.
The robot spawns at X=0 and is scored on total forward distance (X=9 is perfect).
"""

from darwinbot.scenarios import ScenarioInfo

# Zone definitions: (x_start, length, lateral_friction, spinning_friction,
#                    contact_damping, contact_stiffness, restitution, rgba)
ZONES = [
    # Wood: polished, slippery
    (0.0, 3.0, 0.35, 0.02, 10,   1000, 0.05, (0.72, 0.52, 0.30, 1)),
    # Concrete: rough, grippy
    (3.0, 3.0, 1.20, 0.10, 50,   5000, 0.02, (0.55, 0.55, 0.55, 1)),
    # Carpet: very grippy, absorbs energy
    (6.0, 3.0, 2.50, 0.30, 200,  8000, 0.00, (0.40, 0.20, 0.50, 1)),
]

ZONE_WIDTH  = 4.0   # Y extent of each zone
ZONE_HEIGHT = 0.05  # thickness of floor slab


def _make_zone(x_start, length, lat_fric, spin_fric,
               damping, stiffness, restitution, rgba, client):
    import pybullet as p
    half_x = length / 2
    half_y = ZONE_WIDTH / 2

    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half_x, half_y, ZONE_HEIGHT / 2],
        physicsClientId=client,
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_x, half_y, ZONE_HEIGHT / 2],
        rgbaColor=rgba,
        physicsClientId=client,
    )
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[x_start + half_x, 0.0, -ZONE_HEIGHT / 2],
        physicsClientId=client,
    )
    p.changeDynamics(
        body, -1,
        lateralFriction=lat_fric,
        spinningFriction=spin_fric,
        contactDamping=damping,
        contactStiffness=stiffness,
        restitution=restitution,
        physicsClientId=client,
    )
    return body


def build(client: int) -> ScenarioInfo:
    import pybullet as p
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=client)

    for zone in ZONES:
        _make_zone(*zone, client)

    # Invisible boundary walls so the robot doesn't fall off sideways
    wall_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[4.5, 0.05, 0.3],
        physicsClientId=client,
    )
    for y_sign in (+1, -1):
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=wall_col,
            basePosition=[4.5, y_sign * (ZONE_WIDTH / 2 + 0.05), 0.25],
            physicsClientId=client,
        )

    return ScenarioInfo(
        name="surfaces",
        spawn_pos=(0.0, 0.0, 0.35),
        ground_friction=1.02,   # rough average of the three zones
    )

"""
scenarios/confined.py
---------------------
Phase 4 — confined room ("elevator shaft").

A small rectangular room with one opening on one wall.
The robot spawns just outside the opening, must enter,
navigate inside, and exit back through the same opening.

Room dimensions (internal):
  Width  (X): 1.0 m
  Depth  (Y): 1.2 m
  Height (Z): 0.8 m
  Opening:    0.5 m wide × 0.6 m tall, centred on the front wall (Y=0)

Fitness for this scenario is computed differently (see fitness.py):
  - Reward for crossing the threshold INTO the room
  - Additional reward for reaching the back wall
  - Additional reward for exiting back through the opening

The ScenarioInfo.target_pos marks the centre of the room interior.
"""

from darwinbot.scenarios import ScenarioInfo

# Room geometry
ROOM_W   = 1.0    # internal X width
ROOM_D   = 1.2    # internal Y depth
ROOM_H   = 0.8    # internal Z height
WALL_T   = 0.05   # wall thickness
OPENING_W = 0.50  # opening width (centred in X)
OPENING_H = 0.60  # opening height (from floor)

WALL_FRICTION = 0.5
FLOOR_FRICTION = 0.7

# The room's front wall sits at Y=0; interior extends to Y=ROOM_D
# Robot spawns at Y < 0, facing +Y


def _static_box(half_extents, pos, rgba, client):
    import pybullet as p
    col = p.createCollisionShape(p.GEOM_BOX,
                                 halfExtents=half_extents,
                                 physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_BOX,
                              halfExtents=half_extents,
                              rgbaColor=rgba,
                              physicsClientId=client)
    body = p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos,
                             physicsClientId=client)
    p.changeDynamics(body, -1,
                     lateralFriction=WALL_FRICTION,
                     physicsClientId=client)
    return body


def build(client: int) -> ScenarioInfo:
    import pybullet as p
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=client)

    hw = ROOM_W / 2          # half width in X
    ht = WALL_T / 2
    hh = ROOM_H / 2

    # ── Floor ─────────────────────────────────────────────────────────────────
    floor = _static_box(
        half_extents=[hw + WALL_T, ROOM_D / 2 + WALL_T, ht],
        pos=[0.0, ROOM_D / 2, -ht],
        rgba=[0.80, 0.80, 0.80, 1],
        client=client,
    )
    p.changeDynamics(floor, -1,
                     lateralFriction=FLOOR_FRICTION,
                     physicsClientId=client)

    # ── Ceiling ───────────────────────────────────────────────────────────────
    _static_box(
        half_extents=[hw + WALL_T, ROOM_D / 2 + WALL_T, ht],
        pos=[0.0, ROOM_D / 2, ROOM_H + ht],
        rgba=[0.85, 0.85, 0.90, 0.4],
        client=client,
    )

    # ── Back wall (Y = ROOM_D) ────────────────────────────────────────────────
    _static_box(
        half_extents=[hw + WALL_T, ht, hh],
        pos=[0.0, ROOM_D + ht, hh],
        rgba=[0.70, 0.70, 0.75, 1],
        client=client,
    )

    # ── Left wall (X = -hw) ───────────────────────────────────────────────────
    _static_box(
        half_extents=[ht, ROOM_D / 2, hh],
        pos=[-hw - ht, ROOM_D / 2, hh],
        rgba=[0.70, 0.70, 0.75, 1],
        client=client,
    )

    # ── Right wall (X = +hw) ──────────────────────────────────────────────────
    _static_box(
        half_extents=[ht, ROOM_D / 2, hh],
        pos=[hw + ht, ROOM_D / 2, hh],
        rgba=[0.70, 0.70, 0.75, 1],
        client=client,
    )

    # ── Front wall with opening ───────────────────────────────────────────────
    # The front wall (Y=0) has a centred opening of OPENING_W × OPENING_H.
    # Built from three pieces: left pillar, right pillar, top transom.

    open_hw = OPENING_W / 2   # half-width of opening

    # Left pillar
    left_w = hw - open_hw
    if left_w > 0:
        _static_box(
            half_extents=[left_w / 2, ht, hh],
            pos=[-open_hw - left_w / 2, -ht, hh],
            rgba=[0.70, 0.70, 0.75, 1],
            client=client,
        )

    # Right pillar
    if left_w > 0:
        _static_box(
            half_extents=[left_w / 2, ht, hh],
            pos=[open_hw + left_w / 2, -ht, hh],
            rgba=[0.70, 0.70, 0.75, 1],
            client=client,
        )

    # Top transom (above opening, up to ceiling)
    transom_h = ROOM_H - OPENING_H
    if transom_h > 0:
        _static_box(
            half_extents=[open_hw, ht, transom_h / 2],
            pos=[0.0, -ht, OPENING_H + transom_h / 2],
            rgba=[0.70, 0.70, 0.75, 1],
            client=client,
        )

    # ── Approach floor outside the room ───────────────────────────────────────
    _static_box(
        half_extents=[hw, 0.6, ht],
        pos=[0.0, -0.6, -ht],
        rgba=[0.75, 0.75, 0.75, 1],
        client=client,
    )

    return ScenarioInfo(
        name="confined",
        # Spawn just outside the opening, facing +Y (into the room)
        spawn_pos=(0.0, -0.5, 0.35),
        spawn_orn=(0.0, 0.0, 0.0),
        # Target: centre of the room interior
        target_pos=(0.0, ROOM_D / 2, 0.2),
        ground_friction=FLOOR_FRICTION,
    )

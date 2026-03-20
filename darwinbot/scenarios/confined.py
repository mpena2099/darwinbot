"""
scenarios/confined.py
---------------------
Phase 4 — small confined room with a single opening.

Room (internal):
  Width  (X): 1.0 m
  Depth  (Y): 1.2 m
  Height (Z): 0.8 m
  Opening:    0.5 m wide × 0.6 m tall, centred on front wall (Y=0)

Robot spawns just outside the opening, facing +Y (into the room).
Fitness rewards entering, reaching the back wall, and exiting.
"""

from darwinbot.scenarios import ScenarioInfo

ROOM_W    = 1.0
ROOM_D    = 1.2
ROOM_H    = 0.8
WALL_T    = 0.05
OPEN_W    = 0.50
OPEN_H    = 0.60


def get_xml() -> str:
    hw   = ROOM_W / 2
    ht   = WALL_T / 2
    hh   = ROOM_H / 2
    ohw  = OPEN_W / 2       # half opening width
    pillar_w = (hw - ohw)   # width of each door pillar

    transom_h = ROOM_H - OPEN_H

    pillar_xml = ""
    if pillar_w > 0:
        pillar_xml = f"""
    <!-- Front wall: left pillar -->
    <geom name="conf_pillar_l" type="box"
          pos="{-(ohw + pillar_w/2):.4f} {-ht:.4f} {hh:.4f}"
          size="{pillar_w/2:.4f} {ht:.4f} {hh:.4f}"
          rgba="0.70 0.70 0.75 1" condim="6"/>
    <!-- Front wall: right pillar -->
    <geom name="conf_pillar_r" type="box"
          pos="{ohw + pillar_w/2:.4f} {-ht:.4f} {hh:.4f}"
          size="{pillar_w/2:.4f} {ht:.4f} {hh:.4f}"
          rgba="0.70 0.70 0.75 1" condim="6"/>"""

    transom_xml = ""
    if transom_h > 0:
        transom_xml = f"""
    <!-- Front wall: transom above opening -->
    <geom name="conf_transom" type="box"
          pos="0 {-ht:.4f} {OPEN_H + transom_h/2:.4f}"
          size="{ohw:.4f} {ht:.4f} {transom_h/2:.4f}"
          rgba="0.70 0.70 0.75 1" condim="6"/>"""

    return f"""
    <!-- Confined room floor -->
    <geom name="conf_floor" type="box"
          pos="0 {ROOM_D/2:.4f} {-ht:.4f}"
          size="{hw+WALL_T:.4f} {ROOM_D/2+WALL_T:.4f} {ht:.4f}"
          friction="1.2 0.05 0.001"
          rgba="0.80 0.80 0.80 1" condim="6"/>

    <!-- Ceiling -->
    <geom name="conf_ceil" type="box"
          pos="0 {ROOM_D/2:.4f} {ROOM_H+ht:.4f}"
          size="{hw+WALL_T:.4f} {ROOM_D/2+WALL_T:.4f} {ht:.4f}"
          rgba="0.85 0.85 0.90 0.4" condim="6"/>

    <!-- Back wall -->
    <geom name="conf_back" type="box"
          pos="0 {ROOM_D+ht:.4f} {hh:.4f}"
          size="{hw+WALL_T:.4f} {ht:.4f} {hh:.4f}"
          rgba="0.70 0.70 0.75 1" condim="6"/>

    <!-- Left wall -->
    <geom name="conf_left" type="box"
          pos="{-hw-ht:.4f} {ROOM_D/2:.4f} {hh:.4f}"
          size="{ht:.4f} {ROOM_D/2:.4f} {hh:.4f}"
          rgba="0.70 0.70 0.75 1" condim="6"/>

    <!-- Right wall -->
    <geom name="conf_right" type="box"
          pos="{hw+ht:.4f} {ROOM_D/2:.4f} {hh:.4f}"
          size="{ht:.4f} {ROOM_D/2:.4f} {hh:.4f}"
          rgba="0.70 0.70 0.75 1" condim="6"/>
    {pillar_xml}
    {transom_xml}

    <!-- Approach floor outside room -->
    <geom name="conf_approach" type="box"
          pos="0 -0.6 {-ht:.4f}" size="{hw:.4f} 0.6 {ht:.4f}"
          friction="1.2 0.05 0.001"
          rgba="0.75 0.75 0.75 1" condim="6"/>
"""


def build(model, data) -> ScenarioInfo:
    return ScenarioInfo(
        name="confined",
        spawn_pos=(0.0, -0.5, 0.35),
        spawn_orn=(0.0, 0.0, 0.0),
        target_pos=(0.0, ROOM_D / 2, 0.2),
        ground_friction=1.2,
    )

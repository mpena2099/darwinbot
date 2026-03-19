"""
scenarios/flat.py
-----------------
Phase 1 — baseline flat floor.
Already used in the original simulator; wrapped here for consistency.
"""

from darwinbot.scenarios import ScenarioInfo


def build(client: int) -> ScenarioInfo:
    import pybullet as p
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=client)
    plane = p.loadURDF("plane.urdf", physicsClientId=client)

    # Standard friction
    p.changeDynamics(plane, -1,
                     lateralFriction=0.7,
                     restitution=0.1,
                     physicsClientId=client)

    return ScenarioInfo(
        name="flat",
        spawn_pos=(0.0, 0.0, 0.35),
        ground_friction=0.7,
    )

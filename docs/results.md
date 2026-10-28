# Results

A 30-batch run on a 4-core laptop (`--terrains terrain --batches 30 --workers 3`,
when that set was flat, rough and steps;
about 84,000 episodes in 13 hours) filled 98 of the ~500 cells, every one of
them buildable:

| Gait | Types | Best fitness |
|------|-------|--------------|
| `roll` | 35 | 1.03 |
| `walk` | 12 | 0.25 |
| `trot` | 28 | 0.24 |
| `belly` | 14 | 0.16 |
| `crawl` | 9 | 0.13 |

64 types move forward (fitness above 0.05), 30 of them with legs carrying most
of the ground contact. Passive modules are common: 22 elites use a TPU spring,
6 a tail, 4 a caster, 2 a skid and 1 a spike. The most efficient roller has a
cost of transport of 1.6, the most efficient legged robot 1.8. The fastest
robots are still wheeled, and the best one (1.03) appeared by batch 3; the
later batches mostly widened the map. Link lengths stayed between 40 and
125 mm, well inside the 40 to 200 mm the genome allows.

Earlier versions ran 6 batches on the same terrains, before the physics review
that fixed hidden simulator resets and passive joint friction, so the
comparison with this run is only rough:

| | Leg pairs | Tree bodies | Tree + cost of transport | Free form | Free form, 30 batches |
|---|---|---|---|---|---|
| Robot types | 20 | 50 | 51 | 46 | 98 |
| Types that move forward | 14 | 25 | 21 | 28 | 64 |
| Best rolling robot | 1.27 | 0.95 | 0.91 | 1.01 | 1.03 |
| Best legged robot | 0.40 | 0.39 | 0.25 | 0.15 | 0.25 |

![Evolved body plans](img/robots.png)

![MAP-Elites archive](img/archive_map.png)

![Training progress](img/progress.png)

![Printable parts of the best robot](img/parts.png)

## What counts as success (honestly)

| Level | What it means | Promised? |
|-------|---------------|-----------|
| **Minimum** | Diverse robots that walk or roll in simulation, built only from real parts | Yes |
| **Good** | An elite that passes `validate.py` (keeps most of its distance on randomised physics) and exports a coherent kit | Yes, with a full training run |
| **Moonshot** | A printed and assembled robot that walks with its evolved gait | No, it needs the hardware |

The value is in the pipeline being correct end to end, from genome to parts
list, not in any single robot.

## Known limitations

- Wheels still lead on speed: the best rolling robot covers about 6 times the
  distance of the best legged one in a short run, and the top of the archive
  is mostly `roll`. Legged gaits match wheels on efficiency.
- The firmware is half done: `firmware/core/` runs the controller math in portable C++ and a test proves it matches the Python controller tick by tick (`tests/test_firmware_parity.py`). The ESP32 layer (servo bus, load reading, calibration) waits for real hardware.
- Randomised training (`--variants`) is CPU only.
- The only feedback is the servo load. The controller never sees the world, so
  tasks that need steering or reacting (finding a doorway, avoiding a wall) are
  out of scope.
- A body whose parts overlap at spawn, whose parts exceed the 220 mm print bed
  or whose links bend beyond 25 MPa under the servo stall torque is not
  buildable and scores `UNSTABLE_SCORE` without being simulated (common among
  random trees).
- Even after 30 batches the continuous genes stay near the starting sizes
  (links of 40 to 125 mm), and legged gaits lag far behind wheels. Spring stiffness comes from a
  TPU modulus that is not verified, and the printability stress check is a
  cantilever estimate, not a finite-element analysis.

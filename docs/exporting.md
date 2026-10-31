# Exporting a robot

`experiments/export.py --type <type>` writes `exports/<type>/`:

| File | Content |
|------|---------|
| `control.json` | the evolved controller for the ESP32 firmware (format 3), described in [formato-controle.md](formato-controle.md) |
| `bom.md`, `bom.csv` | servos, printed parts with quantities and material, electronics, screws, tyres |
| `assembly.md` | one row per servo in layout order: id, node slot, parent chain in words (for example `core right-front > joint pitch > joint roll`), axis, link length, sign; mounting direction and calibration step |
| `parts/*.stl` | one file per distinct size (needs the `cad` extra): `link_{L}x{W}x{T}.stl` (bent links add `_L{deg}` or `_arc{deg}`), `elbow_{D}mm_{A}deg_{W}x{T}.stl`, `foot_{R}.stl` (blade, pad and claw feet: `foot_{shape}_{R}.stl`), `wheel_{R}.stl`, `spring_{W}x{B}.stl`, `tail_{L}.stl`, `skid.stl`, `spike.stl`, `caster_wheel_{R}.stl`, `caster_fork.stl`, `servo_bracket.stl`, `segment_plate_{L}x{W}.stl` for each spine segment, the torso as `torso_plate.stl` or `torso_tile_{i}.stl`, and `splice_plate_{span}.stl` across each torso cut (sizes in whole millimetres) |
| `robot.xml` | the MuJoCo model, for inspection |

The export walks the module tree: every joint, link and spring needs a link of
its own length and section, every elbow an elbow plate for its distance and
angle, every servo one bracket, every wheel, caster, tail, skid, spike and foot
(explicit, or the free tip of a limb) one print. Links, elbows, feet and springs
get one STL per distinct size. Springs and spikes are printed in TPU, the rest in
PLA. The bill of materials counts everything from the tree.

`python tools/parts_sheet.py exports/<type>` draws every STL of the export on
one image (`exports/<type>/parts.png`), with its bounding box in mm, to check
the parts before slicing.

## Hollowed parts (optional)

`experiments/export.py --type <type> --organic` also writes `parts_organic/`:
the links and elbow plates hollowed by 2D topology optimisation
(`darwinbot/organic.py`), with the horn end clamped and the stall load at the
far end. Evolution chose each part's size and interfaces; the optimisation only
removes material from the middle, keeping a rim along every edge and a solid
boss at both ends, so holes and outline are unchanged. Parts typically keep 50
to 90% of their material, and each takes about 10 s.

Two cautions: the load case is in-plane only (out-of-plane bending is carried by
the rim, not optimised), and the hollowed robot is lighter than the one
evolution scored. Nothing here has been printed yet.

Servo dimensions in `darwinbot/parts.py` (horn and mounting holes, body size)
are placeholders marked "measure a real STS3215 and update": check them
against a real servo before printing.

# Exporting a robot

`experiments/export.py --type <type>` writes `exports/<type>/`:

| File | Content |
|------|---------|
| `control.json` | the evolved controller for the ESP32 firmware (format 4), described in [formato-controle.md](formato-controle.md) |
| `bom.md`, `bom.csv` | servos, printed parts with quantities and material, electronics, screws, tyres |
| `assembly.md` | one row per servo in layout order: id, node slot, parent chain in words (for example `core right-front > joint pitch > joint roll`), axis, link length, sign; mounting direction and calibration step |
| `parts/*.stl` | one file per distinct size (needs the `cad` extra): `link_{L}x{W}x{T}.stl` (bent links add `_L{deg}` or `_arc{deg}`; a joint's link adds `_u{axis}` for the fork, a tip cradle `_c{axis}`), `elbow_{D}mm_{A}deg_{W}x{T}.stl`, `foot_{R}.stl` (blade, pad and claw feet: `foot_{shape}_{R}.stl`), `wheel_{R}x{W}.stl`, `spring_{W}x{B}.stl`, `tail_{L}.stl`, `skid.stl`, `spike.stl`, `caster_wheel_{R}.stl`, `caster_fork.stl`, `cradle_{axis}.stl` (variants `_w{deg}`, `_back`, `_free`) with `wedge_{deg}.stl`, `segment_plate_{L}x{W}.stl` for each spine segment, the torso as `torso_plate.stl` or `torso_tile_{i}.stl`, and `splice_plate_{span}.stl` across each torso cut (sizes in whole millimetres) |
| `robot.xml` | the MuJoCo model, for inspection |

The export walks the module tree: every joint, link and spring needs a link of
its own length and section, every elbow an elbow plate for its distance and
angle, every servo one cradle, every wheel, caster, tail, skid, spike and foot
(explicit, or the free tip of a limb) one print. Links, elbows, feet and springs
get one STL per distinct size. Springs and spikes are printed in TPU, the rest in
PLA. The bill of materials counts everything from the tree.

## Cradle, U link and wedge

A servo is held by a printed cradle that belongs to its parent, around the body
as the simulation places it (0.4 mm of slack per side, an M2 cap on each horn
face with the four screws into the tapped holes). On the torso the cradle is its
own part (`cradle_{axis}.stl`) with a foot carrying the common M3 pattern; at the
tip of a link it is printed as one piece with the link, and an elbow plate ends
in the cradle of its child. The joint's own link starts as a U fork that clasps
both horns of the servo, one arm on each horn face with the horn hole pattern,
and joins the plate beyond the servo, out of the circle the cradle fills, so the
link turns the whole joint range.

The foot goes down to the torso plate on the side that points down. When the slot
is rolled by an angle that is not a multiple of 90 degrees (up to 20), the foot
is tilted and slid along the axis, and a printed `wedge_{deg}.stl` between the tab
and the foot takes up the tilt; face mounts and segment plates never get one. A
servo whose output axis stands vertical bolts down behind itself (`_back`). A face
mount, a roll joint whose body would stand through the plate, or a roll steeper
than 20 degrees gets a cradle without a foot (`_free`) that you have to hold with
a bracket of your own.

Parts that are rigid with a link and overlap it are printed as one piece with it:
a plain link hung on a joint or on another link (`_k{n}` in the name), the bar to
a passive module on the side of a link (`_b{distance}a{angle}`, ending in a spigot
for a foot), and the cradle of the servo at the tip. The side servos hang on
elbow plates bolted to a fin the link tip narrows to; the fin and the cut beside
it follow the cradle's distance, angle and roll (`_e{at}x{x0}x{x1}`). The torso
plate sits `plate_drop` mm below the joint plane, 22 or more when a servo on an
edge hangs lower. Everything is drawn in whole millimetres and degrees, so a part
can sit half a millimetre from the simulation's frame; the check places the
modules where the printed parts put them.

`python tools/check_assembly.py --archive a.pkl --type <type>` builds every part at
its simulation frame and reports each pair of printed parts and servo bodies that
overlap, at the spawn pose and at four poses of a straight walk (0.5, 1.5, 2.5 and
3.5 s). Only the screwed laps (0.5 mm or less: a cradle on its servo, an elbow
tongue on its link, a wedge on its tab) are allowed. The simulation collides the
same shapes (cradle, fork, bosses, elbow strut, torso plate and tabs), so a body
it accepts is meant to pass this check.

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
come from the SO-ARM100 STEP model of the STS3215, not from a caliper
measurement: check them against a real servo before printing.

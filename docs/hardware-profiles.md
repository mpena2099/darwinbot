# Hardware profiles

A hardware profile is a **kit**: a small YAML file that names one component for each of five sections. The default kit, `darwinbot/profiles/kits/default.yaml`, describes the documented build: Feetech STS3215 12 V servos, a 3S LiPo, an ESP32, PLA links and TPU feet.

```yaml
servo: sts3215-12v
battery: 3s-1200mah
electronics: esp32
material: pla-tpu
printer: bed-220mm
```

## Components

Each component is a YAML file with the keys of its section, and the numbers carry their source in comments. The shipped ones live in `darwinbot/profiles/<section plural>/`.

- `servos/`: no-load speed, stall torque, the load that reads as `N = 1`, mass, cradle masses (on a plate and at a link), idle and stall current, joint friction, armature and gain, wheel gain, the simulation box, and the housing and horn measures that `parts.py` and the exporter use (body size, output axis offset, clearance, tapped holes, bolt circle).
- `batteries/`: capacity, C rating and voltage.
- `electronics/`: mass, current and footprint of the board and pack, plus the board itself: its BOM name and note, the hole pattern in the torso plate, the hole diameter and the screw size.
- `materials/`: densities, PLA strength, the safety factor on link bending and the friction factor of hard plastic on the floor.
- `printers/`: the largest part that fits the bed.

A kit entry may be:

- a component name, looked up first in a folder of that section next to the kit file (`./servos/<name>.yaml`), then in the shipped folder;
- a path to a component file, relative to the kit file (`servo: parts/my-servo.yaml`);
- an inline mapping with the whole section, which is how profiles were written when they were a single file, so an old profile still loads unchanged.

A missing component, or a component with a missing or unknown key, stops the program with a message that names the section and the file.

## Your own board

The board is a component like the servo. To use another one, copy `darwinbot/profiles/electronics/esp32.yaml`, name it in a kit (`electronics: my-board`) and edit:

| Key | Meaning |
|---|---|
| `mass`, `current_a`, `box` | everything on the torso besides the servos: board, regulator, bus adapter and pack; the footprint in metres |
| `board_name`, `board_note` | the BOM row of the board and its note |
| `board_label` | what the board's screws are listed for in the BOM |
| `holes` | the hole pattern in mm, centred behind the origin of the plate: `[width, height]` for a rectangle, or a list of `[x, y]` points |
| `hole_d` | clearance hole in the torso plate, mm |
| `screw_d` | screw diameter, mm; one screw per hole. `3.0` joins the M3 screws of the joints in the BOM, any other size gets its own screw and nut rows |

The five keys after `box` are optional. A profile without them, such as the one an old archive stores, reads as the ESP32 board with a 58 x 23 mm pattern, so it loads and exports as before. `tools/check_kit.py` also warns when a hole is outside the footprint or smaller than its screw.

`examples/kits/pi-zero-2w/` is a worked example for a Raspberry Pi Zero 2 W with a 5 V regulator and a half-duplex bus adapter. Its masses and currents are labelled as estimates in the file.

The board changes the torso plate and the BOM. The program that runs on it is the firmware core plus a platform layer (`firmware/README.md`), and that layer exists for no board today.

## Using another kit

To try another servo, copy the default kit, put the new servo next to it and name it:

```bash
cp darwinbot/profiles/kits/default.yaml my-kit.yaml
mkdir servos
cp darwinbot/profiles/servos/sts3215-12v.yaml servos/my-servo.yaml   # edit the numbers
sed -i 's/^servo: .*/servo: my-servo/' my-kit.yaml

python experiments/train.py --profile my-kit.yaml --preset steer
python tools/kaggle/islands.py ... --profile my-kit.yaml
```

`DARWINBOT_PROFILE=my-kit.yaml` does the same for any command. The kit's name (its file name without the extension) is what the logs show. The islands tool resolves the kit locally and sends the whole resolved profile to the Kaggle kernels, so the components next to your kit reach them.

## Checking a kit

Before a run, check the numbers:

```bash
python tools/check_kit.py my-kit.yaml
```

It reports every missing or unknown field, and every value outside a plausible range for its unit: a mass or a speed of zero, a body length given in metres (`length: 0.045`) or a simulation box given in millimetres (`box: 45`). The cross checks catch a stall current below the idle one, a servo box that disagrees with the housing length or width, and a battery voltage far from the 12 V the servo name suggests. It also prints the fields that differ from the default kit (old -> new) and the derived numbers: usable torque, runtime at idle and at 30 % of stall, the mass of an 8-servo robot and the bending stress of the longest link. Warnings are unusual but possible; errors exit with code 1, and `train.py --profile` prints the warnings and refuses to start on an error.

## The archive remembers it

An archive stores the profile it was trained on. `train.py --resume`, `replay.py`, `validate.py`, `export.py`, `snapshot.py`, `polish.py` and `prune.py` read it back and, when it differs from the active one, restart themselves under the archive's profile. Archives from before profiles count as the default profile. `merge_archives.py` refuses archives with different profiles, and `train.py --resume` refuses a `--profile` that differs from the archive's.

## Limits

A profile changes numbers, not shapes. The servo must keep the STS3215 housing layout (output axis along the body, horns on both faces, the same tapped-hole pattern), and the printed parts in `parts.py` keep their geometry: cradles, forks and elbows are drawn around the housing measures but their other dimensions are fixed. A servo with a different shape needs changes in `parts.py` and `morphology.py`. Changing hardware values changes the physics, so archives trained on different profiles are not comparable.

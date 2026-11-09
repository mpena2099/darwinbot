# Hardware profiles

A profile is one YAML file with every hardware number the project uses. The default, `darwinbot/profiles/sts3215-12v.yaml`, describes the documented build: Feetech STS3215 12 V servos, a 3S LiPo, an ESP32, PLA links and TPU feet.

## What a profile holds

- `servo`: no-load speed, stall torque, the load that reads as `N = 1`, mass, bracket mass, idle and stall current, joint friction, armature and gain, wheel gain, the simulation box, and the housing and horn measures that `parts.py` and the exporter use (body size, output axis offset, clearance, tapped holes, bolt circle).
- `battery`: capacity, C rating and voltage.
- `electronics`: mass, current and footprint of the board and pack.
- `material`: densities, PLA strength, the safety factor on link bending and the friction factor of hard plastic on the floor.
- `printer`: the largest part that fits the bed.

A missing or unknown key stops the program with a message that names it.

## Using another profile

```bash
cp darwinbot/profiles/sts3215-12v.yaml my-servo.yaml     # edit the numbers
python experiments/train.py --profile my-servo.yaml --preset steer
python tools/kaggle/islands.py ... --profile my-servo.yaml
```

`DARWINBOT_PROFILE=my-servo.yaml` does the same for any command.

## The archive remembers it

An archive stores the profile it was trained on. `train.py --resume`, `replay.py`, `validate.py`, `export.py`, `snapshot.py`, `polish.py` and `prune.py` read it back and, when it differs from the active one, restart themselves under the archive's profile. Archives from before profiles count as the default profile. `merge_archives.py` refuses archives with different profiles, and `train.py --resume` refuses a `--profile` that differs from the archive's.

## Limits

A profile changes numbers, not shapes. The servo must keep the STS3215 housing layout (output axis along the body, horns on both faces, the same tapped-hole pattern), and the printed parts in `parts.py` keep their geometry: cradles, forks and elbows are drawn around the housing measures but their other dimensions are fixed. A servo with a different shape needs changes in `parts.py` and `morphology.py`. Changing hardware values changes the physics, so archives trained on different profiles are not comparable.

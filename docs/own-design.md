# Starting from your own design

Evolution does not have to start from random bodies. You can write a robot by
hand in a YAML file, seed an archive with it, and let `train.py --resume`
improve it. Parts you mark `lock: true` are kept as you wrote them.

```bash
python experiments/seed.py --robot examples/robots/two-wheel-car.yaml --out checkpoints/car
python experiments/train.py --resume checkpoints/car/archive.pkl --batches 30
```

`seed.py` optimises the controller of your body, puts it in a new archive and
prints the `train.py --resume` command to continue. It never overwrites an
existing `archive.pkl`. If the body is not buildable (a printing rule, or parts
that overlap at the start), it lists what failed and stops; it does not repair
the body for you.

## The file

Units are millimetres and degrees. Attachment fractions (`offset`, `angle`) stay
between 0 and 1.

```yaml
name: two-wheel-car
torso: {length_mm: 160, width_mm: 120, contour: [1,1,1,1,1,1,1,1], lock: true}
nodes:
  left_wheel:  {parent: torso, slot: 2, kind: wheel, radius_mm: 60, width_mm: 20, angle: 0.25, lock: true}
  right_wheel: {parent: torso, slot: 5, kind: wheel, radius_mm: 60, width_mm: 20, angle: 0.75, lock: true}
  caster:      {parent: torso, slot: 1, kind: caster}
```

Nodes are listed parent first. A node needs `parent` (`torso` or an earlier node
name), `slot` and `kind`; everything else takes a default. Node ids (the stable
slot ids of the genome) are given from 1 in file order, or by an `id:` key,
which `--dump` writes so a reloaded robot keeps its ids.

| Kind | Fields |
|------|--------|
| `joint` | `axis` (pitch, yaw, roll), `flip`, `length_mm`, `width_mm`, `thick_mm`, `profile` (straight, L, arc), `bend_deg` |
| `link` | `length_mm`, `width_mm`, `thick_mm`, `profile`, `bend_deg` |
| `spring` | `flip`, `length_mm`, `width_mm`, `thick_mm` (the TPU blade), `profile`, `bend_deg` |
| `segment` | `axis`, `flip`, `length_mm`, `width_mm` |
| `wheel` | `radius_mm`, `width_mm` |
| `caster` | `radius_mm` |
| `foot` | `radius_mm`, `shape` (sphere, blade, pad, claw) |
| `tail` | `length_mm`, `tail_mass_g` |
| `skid`, `spike` | none |

Every node also accepts `roll_deg` (turn around the slot direction), `offset`,
`angle` and `lock`. The torso takes `length_mm`, `width_mm`, `contour` (8
factors, 0.4 to 1.4) and `lock`.

Every value is checked against the genome ranges (the same ones evolution uses),
and a bad one raises an error that names the node and the field and gives the
allowed range. Unknown keys, unknown parents, a slot used twice, more than 20
nodes (the torso counts) and a field that does not belong to the kind are
refused as well. Finally the file must survive `decode(encode(body))` unchanged:
that catches the limits of 16 servos, depth 5 and the segment count.

## Seeing your robot

To check the shape while you write the file, open it in the viewer. It stands
still on the ground, with every joint at the middle of its range and the wheels
stopped, since it has no controller yet:

```bash
python experiments/replay.py --robot examples/robots/small-walker.yaml
```

It prints a summary of the body and any printing rule it breaks or parts that
overlap at the start (`seed.py` lists which ones). `--terrain` picks the ground
and `--record out.mp4` saves a video instead of opening the window.

To see it move, give it a controller first: a short `seed.py` run is enough for
a first look, then replay the robot from the archive it wrote:

```bash
python experiments/seed.py --robot examples/robots/small-walker.yaml --budget 50 --out checkpoints/walker
python experiments/replay.py --archive checkpoints/walker/archive.pkl --list
python experiments/replay.py --archive checkpoints/walker/archive.pkl --type <type from the list>
```

## Slots

A slot is a place on the parent where a child attaches.

- **Torso** (12 slots). Contour: 0 front, 1 back, 2 to 4 left side (front corner,
  middle, rear corner), 5 to 7 right side (front corner, middle, rear corner).
  Faces: 8 top front, 9 top back, 10 bottom front, 11 bottom back (the child
  points up or down; a servo cannot sit on a face). On a contour slot `offset`
  slides the mount along the edge by up to 30% and `angle` yaws it by up to
  60 degrees (0.5 is straight out). On a face slot `offset` slides it across the
  torso and `angle` yaws it about the vertical.
- **Segment** (8 slots): the same contour numbering on its own plate. Slot 1 faces
  its parent and is never used. A segment attaches only to slot 0 or 1 of the
  torso or of another segment.
- **Joint, link, spring** (3 slots): 0 is the tip (the end of the link path);
  1 and 2 are the left and right side of an elbow at the end of the link. There
  `offset` sets the elbow distance (45 to 80 mm) and `angle` the elbow angle
  (30 to 150 degrees, 0.5 is 90).
- **Wheel, caster, foot, tail, skid, spike** have no slots: nothing attaches to them.

## Editing an evolved robot

```bash
python experiments/seed.py --dump --archive checkpoints/archive.pkl --type 8servos-low-crawl > mine.yaml
```

writes the elite's body as the same YAML (without its controller). Edit it, add
`lock: true` where you want, and seed a new archive from it.

## What lock means

`lock: true` on a node keeps its numbers (kind, size, roll, mount fractions, and
its parent and slot) through mutation and grafting. On the torso it keeps the
length, width and contour. During a run, `evolve.run` redraws an offspring up to
8 times until the locked parts are unchanged, otherwise the parent's body is
searched again with a new controller. Random bodies are not drawn at all in an
archive with locks, since they would not carry the parts.

Limits:

- Locks need a single `--robot`; `seed.py` refuses them with several robots.
- The archive still keeps the best robot per cell, so a locked part stays in
  every robot, but your original design can be replaced in its cell by a better
  variant that has the same locked parts.
- A lock keeps the part's numbers, not where it ends up in space: if the node
  it hangs from changes (a longer link, a rolled mount), the locked part moves
  with it.
- Seeding several robots (no locks) starts a diverse archive; evolution then
  mutates all of them freely.

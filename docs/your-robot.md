# Constraining the body

Evolution builds whatever walks best, and that is not always the robot you want
to build: it may grow wheels when you want legs, a spine when your printer is
small, or sixteen servos when your budget stops at eight. `--body` limits the
bodies the run may make:

```bash
python experiments/train.py --body "no-wheels,max-servos=8,legs=4..6"
```

The limits apply where bodies are made, not in the score: a forbidden module is
never created, and a body that still breaks a limit is rejected and never
simulated. The score keeps rewarding the same things, so a cheap quadruped
still learns to walk as well as it can under its constraints. The string is
saved in the archive, like `--rules`; `replay`, `validate` and `export` read it
back, and resuming a run keeps the limits it was trained with.

Leave the option empty (the default) to evolve with today's limits.

## Tokens

Pass one or more tokens separated by commas, in any order:

| Token | Default | What it allows |
|-------|---------|----------------|
| `no-wheels` | wheels allowed | No `wheel` modules. |
| `no-passive` | passive modules allowed | No `spring`, `tail`, `skid`, `spike` or `caster`. |
| `no-segments` | spine segments allowed | No `segment` plates (the spine), so every limb hangs on the core. |
| `max-servos=N` | 16 | At most N servos, counting joints, wheels and spine segments. N must be at least 1. |
| `legs=A..B` | any number of legs | Between A and B legs. `legs=N` means exactly N. |
| `max-size=MM` | no limit | Largest side of the body's bounding box at the spawn, in mm. |
| `bed=MM` | 220 | The print bed the printability checks use, in mm; smaller values also make the repair shorten or cut parts. |

A leg is what the archive axis counts: a limb on the torso or a spine plate that
contains a servo joint and no wheel. A wheeled limb is not a leg, and neither is
a spine segment.

Limits are checked twice. Random bodies, added modules, type changes,
duplicates and grafts obey them while they are generated: the search does not
waste an evaluation on a body it would have to throw away. `fit_rules` then
repairs what a mutation still breaks: extra servos are cut with their subtrees
(a wheel becomes a foot, one joint of a many-jointed limb becomes a link), and
legs are added or removed until the count fits. A body that still breaks a
limit is reported by `violations` and is never simulated.

Some combinations leave little or nothing to build. `legs=0` with `no-wheels`
forbids both ways of putting a servo on the ground, and `max-servos=1` with
`legs=4` cannot be satisfied: those runs seed and then stall with no buildable
body. Ask for limits that can meet each other.

## Ready examples

A cheap quadruped: no wheels, at most eight servos, exactly four legs.

```bash
python experiments/train.py --body "no-wheels,max-servos=8,legs=4" --terrains terrain --batches 30
```

Wheels only: no passive modules and no legs, so the servos drive wheels or the
spine instead of limbs.

```bash
python experiments/train.py --body "no-passive,legs=0" --terrains flat --batches 30
```

A small printer: parts must fit a 180 mm bed and the whole body a 300 mm box.

```bash
python experiments/train.py --body "bed=180,max-size=300" --terrains terrain --batches 30
```

The limits do not change the environments, the task or the archive axes; combine
`--body` with `--terrains` and `--payload` as usual. A run with a different
string is a different experiment, so use a separate `--out` folder or a
different preset name for it.

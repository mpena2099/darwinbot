# Fitness and tasks

## Tasks

`--task` picks what the score rewards; the body and controller genomes are the
same for every task, and so are the behaviour descriptors.

- `steer` (default): walk straight and turn when told. A steering command from
  -1 (left) to +1 (right) scales each joint's amplitudes, shifts its offset and
  changes each wheel's speed through two genes per slot, the way animals turn
  with one gait and one side stronger. Every evaluation runs three episodes,
  straight, left and right: `score = 0.5 * straight + 0.25 * (left + right)`,
  where `straight` is the forward score and a turn scores `0.2 * yaw` towards the
  side asked for (`yaw` in radians, counted up to 90 degrees), never more than
  the distance the torso travelled in that episode, minus the same energy and
  fall terms. The robot has to walk a curve: spinning on the spot earns
  nothing. Three
  episodes make an evaluation three times as slow. On the robot the command can
  come from a remote control, and later from sensors.
- `forward`: walk straight only, the score above.
- `turn`: rotate in place. `score = |yaw| - 0.5 * hypot(dx, dy) - 0.05 * energy / (m * g) - 0.5 * fell`,
  where `yaw` is the net heading change of the torso in radians, either way.

`--payload KG` adds mass to the torso payload box for the whole run. The task
and payload are stored in the archive, so `replay.py`, `validate.py` and
`export.py` use the same conditions the robot was trained under (`replay.py
--list` prints them in a header line when they are not the default).

Ready-made tasks run with `--preset`, one after the other, each in its own
folder (`checkpoints/<preset>`), so they never overwrite the main archive:

| Preset | Task | Terrain | Payload |
|---|---|---|---|
| `steer` | walk and turn when told | the `terrain` set | none |
| `forward` | walk forward | the `terrain` set | none |
| `turn` | turn in place | flat | none |
| `step` | walk and turn when told | one 3 cm step | none |
| `payload` | walk and turn when told | the `terrain` set | 0.3 kg |

```bash
# The three extra tasks, 5 batches each. step and payload score three episodes
# per evaluation, so each takes about 10 h on 4 cores; turn is much quicker
python experiments/train.py --preset turn,step,payload --batches 5 --workers 3
# The same command again continues each preset from where it stopped
python experiments/replay.py --archive checkpoints/turn/archive.pkl --list
```

Ctrl+C saves the preset being trained and skips the rest.

## Fitness

Per episode, from `darwinbot/fitness.py`:

```
score = dx - 0.5 * |dy| - 0.05 * energy / (m * g) - 0.5 * fell
```

`dx` is forward displacement, `dy` lateral displacement, `energy` the mechanical
work of all actuators, `m` the robot mass and `g = 9.81`. Dividing energy by
weight turns it into metres and keeps heavy and light bodies comparable. A fall
ends the episode but keeps the distance covered, so the fixed penalty only
breaks ties in favour of staying upright. A physics blow-up returns
`UNSTABLE_SCORE = -5.0`.

Multi-terrain runs use a blend of mean and worst score:

```
fitness = 0.7 * mean(scores) + 0.3 * min(scores)
```

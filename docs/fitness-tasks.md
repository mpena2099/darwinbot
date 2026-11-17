# Fitness and tasks

## Tasks

`--task` picks what the score rewards; the body and controller genomes are the
same for every task, and so are the behaviour descriptors.

- `steer` (default): walk straight, turn and change pace when told. A
  steering command from -1 (left) to +1 (right) scales each joint's amplitudes,
  shifts its offset and changes each wheel's speed through two genes per slot,
  the way animals turn with one gait and one side stronger. A speed command from
  0 to 1 (1 is the gait as evolved) does the same through two more genes: with
  `d = 1 - speed`, amplitudes are multiplied by `1 - d * speed_gain`, offsets
  shift by `d * speed_shift` and wheel speeds by `d * speed_shift`; the
  frequency is shared by the whole body and does not change. Every evaluation
  runs four episodes, straight, left, right and straight at speed 0.5:
  `score = 0.4 * straight + 0.2 * (left + right) + 0.2 * half`, where
  `straight` is the forward score and a turn scores `0.2 * yaw` towards the
  side asked for (`yaw` in radians, counted up to 90 degrees), never more than
  the distance the torso travelled in that episode, minus the same energy and
  fall terms. The robot has to walk a curve: spinning on the spot earns
  nothing. `half` is `0.5 * P - |P_half - 0.5 * P|` minus the energy and fall
  terms, with `P` the upright-weighted progress of the full-speed episode: it
  is best at exactly half the distance and zero for a robot that ignores the
  command. Four episodes make an evaluation four times as slow as a forward-only
  one. On the robot the commands can come from a remote control, a ROS 2
  `Twist`, and later from sensors.
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
# The three extra tasks, 5 batches each. step and payload score four episodes
# per evaluation, so each takes about 10 h on 4 cores; turn is much quicker
python experiments/train.py --preset turn,step,payload --batches 5 --workers 3
# The same command again continues each preset from where it stopped
python experiments/replay.py --archive checkpoints/turn/archive.pkl --list
```

Ctrl+C saves the preset being trained and skips the rest.

## Optional score rules

`train.py --rules a,b` switches on score rules that are off by default; they are
stored in the archive (`replay.py` reads them back), so a short run with one rule
can be compared against the current score. Their metrics are always measured,
in `sim.py` and `sim_gpu.py`.

| Rule | Effect |
|---|---|
| `worse_turn` | steer: `0.5 * straight + 0.5 * min(left, right)` |
| `turn_forward` | steer: a turn earns nothing when the torso ended behind the spawn (`dx <= 0`) |
| `tilt` | every episode: `- 0.2 * tilt`, the mean torso tilt in radians |
| `slip` | every episode: `- 0.2 * slip`, the summed horizontal speed (m/s) of non-wheel terminals that touched the ground on two consecutive control ticks, averaged over ticks |
| `servo_ground` | every episode: `- 0.1 * servo_ground`, the share of control ticks a servo housing touches the ground |

## Fitness

Per episode, from `darwinbot/fitness.py`:

```
score = up(progress) - 0.5 * |dy| - 0.2 * |yaw| - 0.05 * energy / (m * g) - 0.5 * fell
```

`progress` is the forward distance of the second half of the episode, doubled:
`2 * (x_end - x_mid)` along the heading axis, so a steady gait keeps its
distance and a one-time drop of the torso early on is mostly ignored. `up()` scales a
positive progress by the mean cosine of the torso tilt: a robot that leans 45
degrees keeps 71% of it. `dy` is lateral displacement, `yaw` the net heading
change in radians (charged at the rate a turn earns it), `energy` the mechanical
work of all actuators, `m` the robot mass and `g = 9.81`. Dividing energy by
weight turns it into metres and keeps heavy and light bodies comparable. A fall
ends the episode. A fall before the half-way step has no progress at all; a fall
after it keeps `2 * (x_fall - x_mid)`. The fixed penalty breaks ties in favour
of staying upright. A physics blow-up returns `UNSTABLE_SCORE = -5.0`.

Multi-terrain runs use a blend of mean and worst score:

```
fitness = 0.7 * mean(scores) + 0.3 * min(scores)
```

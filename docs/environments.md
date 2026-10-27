# Environments and presets

Every robot is scored on each environment of the run, and its fitness mixes the
average with the worst one (`0.7 * mean + 0.3 * min`), so the search favours
robots that cope with all of them. Pick the environments with `--terrains`:

```bash
python experiments/train.py --terrains rubble            # one environment only
python experiments/train.py --terrains flat,soft,steep   # your own list
python experiments/train.py --terrains terrain           # a ready set (below)
```

Fewer environments make a specialist (the best robot for sand, or for a
staircase); more make a generalist. The choice is saved in the archive, so
`replay`, `validate` and `export` use the same ones.

| Name | What the robot faces |
|------|----------------------|
| `flat` | flat floor |
| `rough` | smoothed random bumps up to 4 cm, flat near the spawn |
| `rubble` | 40 scattered rocks 3 to 7 cm high, taller than many wheels can climb |
| `steps` | staircase of six 3 cm steps from x = 0.6 m |
| `ledges` | staircase of four 6 cm steps, too tall to roll up |
| `step` | one 3 cm step, 0.50 m ahead of the spawn, 1 m deep |
| `slope` | 8 degree ramp up from x = 0.5 m |
| `steep` | the same ramp at 15 degrees |
| `downhill` | spawn on a plateau, then a 10 degree slope down to the floor |
| `slippery` | floor with friction 0.2, like ice |
| `soft` | yielding floor like sand: feet sink 1 to 2 cm and wheels feel rolling drag |

Ready sets:

| Set | Environments |
|-----|--------------|
| `flat` | flat |
| `terrain` | flat, rough, steps, rubble |
| `all` | all of the above except `step` |

`rough` and `rubble` are random. Training draws new bumps and rocks for every
body, so a family of robots cannot memorise one layout; `replay` and `validate`
use one fixed layout, so a robot always meets the same ground there.

## Tasks and presets

The ready-made tasks and the `--preset` option are described in [Fitness and tasks](fitness-tasks.md).

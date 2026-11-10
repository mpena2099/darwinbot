# Plugins: your own tasks and terrains

A plugin is a Python file that adds tasks (what the score rewards) or
terrains (the ground the robot walks on) without changing DarwinBot's code.
Nothing is loaded unless a command names the file with `--plugin`.

## Writing one

```python
# my_plugin.py
from darwinbot import terrain as te
from darwinbot.fitness import register_task


@register_task("backward")
def backward(m):
    """Metres walked backwards, minus a sideways drift penalty."""
    return -m.dx - 0.5 * abs(m.dy)


def ramp():
    return te.Terrain("ramp", world=te.slope(degrees=5).world)


te.register_terrain("ramp", ramp)
```

A task's score function takes one `Metrics` (the measures of one episode)
and returns a float. It can use only the fields the simulator already
measures: `dx`, `dy`, `yaw`, `energy`, `mass`, `fell`, `duration`, `duty`,
`height`, `wheel_share`, `upright`, `progress`, `tilt`, `slip` and
`servo_ground` (see `darwinbot/sim.py`). A physics blow-up still scores
`UNSTABLE_SCORE` before your function is called.

A task that runs several episodes passes `commands`, a tuple of
`(steering, speed)` pairs; its function then gets the list of `Metrics`, one
per command:

```python
register_task("zigzag", lambda ms: min(m.progress for m in ms),
              commands=((-1.0, 1.0), (1.0, 1.0)))
```

A terrain is a function that returns a `Terrain`: the MJCF ground geoms in
`world`, and optionally `assets`, `friction` and `spawn_z` (the ground height
under the spawn). Pass `seeded=True` to `register_terrain` when the factory
takes a `seed` argument and should be drawn anew for every body in training,
like `rough` and `rubble`. Built-in names cannot be replaced.

## Using it

```bash
python experiments/train.py --plugin my_plugin.py --task backward --terrains flat,ramp
python experiments/seed.py --robot my_robot.yaml --plugin my_plugin.py --terrains ramp
```

The archive stores the plugin's file name and the SHA-256 of its bytes. Every
other script (`replay`, `validate`, `export`, `snapshot`, `polish`, `prune`)
needs the same `--plugin` to open that archive:

- without `--plugin`, it stops and names the file the archive was trained with;
- with a file whose bytes differ, it prints a warning and goes on, since an
  edited plugin may score or build the ground differently from the run.

Plugins are ordinary Python and run with your permissions: load only files
you wrote or have read.

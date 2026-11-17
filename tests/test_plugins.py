"""
test_plugins.py
---------------
External tasks and terrains, loaded only through an explicit --plugin file.
"""

import hashlib
import pickle

import pytest

from darwinbot import evolve, fitness, plugins
from darwinbot import terrain as te
from darwinbot.sim import Metrics


PLUGIN = '''
from darwinbot import terrain as te
from darwinbot.fitness import register_task


@register_task("backward")
def backward(m):
    return -m.dx


def bump():
    return te.Terrain("bump", world=te._plane())


te.register_terrain("bump", bump)
'''


@pytest.fixture(scope="module")
def plugin_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("plugin") / "my_plugin.py"
    path.write_text(PLUGIN)
    plugins.load(str(path))
    return path


def _metrics(**kw):
    return Metrics(**{"dx": 1.0, "dy": 0.1, "energy": 2.0, "fell": False,
                      "unstable": False, "duration": 5.0, "mass": 1.0, **kw})


def test_load_registers_task_and_terrain(plugin_path):
    record = plugins.load(str(plugin_path))
    assert record == {"name": "my_plugin.py",
                      "sha256": hashlib.sha256(plugin_path.read_bytes()).hexdigest()}
    assert "backward" in fitness.TASKS
    assert "bump" in te.TERRAINS
    assert te.TERRAINS["bump"]().name == "bump"


def test_builtin_scores_unchanged(plugin_path):
    m = _metrics()
    expected = (m.progress - fitness.LATERAL_W * abs(m.dy)
                - fitness.HEADING_W * abs(m.yaw)
                - fitness.ENERGY_W * m.energy / (m.mass * fitness.G))
    assert fitness.score(m, "forward") == pytest.approx(expected)
    assert fitness.task_score([m], "forward") == fitness.score(m, "forward")
    assert fitness.commands("forward") == fitness.STRAIGHT
    assert fitness.commands("steer") == fitness.COMMANDS["steer"]


def test_plugin_task_dispatches_and_keeps_unstable_guard(plugin_path):
    m = _metrics(dx=1.5)
    assert fitness.score(m, "backward") == pytest.approx(-1.5)
    assert fitness.task_score([m], "backward") == pytest.approx(-1.5)
    assert fitness.commands("backward") == fitness.STRAIGHT
    assert fitness.task_score([_metrics(unstable=True)],
                              "backward") == fitness.UNSTABLE_SCORE


def test_register_terrain_refuses_a_builtin():
    with pytest.raises(ValueError, match="flat"):
        te.register_terrain("flat", te.flat)


def test_require_names_the_plugin_and_the_flag():
    a = evolve.Archive(terrains="flat")
    a.plugin = {"name": "my_plugin.py", "sha256": "0" * 64}
    with pytest.raises(SystemExit) as e:
        plugins.require(a)
    assert "my_plugin.py" in str(e.value) and "--plugin" in str(e.value)


def test_require_warns_on_a_changed_plugin(plugin_path, capsys):
    a = evolve.Archive(terrains="flat")
    a.plugin = {"name": "my_plugin.py", "sha256": "0" * 64}
    plugins.require(a, str(plugin_path))
    out = capsys.readouterr().out
    assert "warning" in out.lower() and "sha256" in out


def test_old_archive_without_plugin_loads():
    a = evolve.Archive(terrains="flat")
    state = dict(a.__dict__)
    state.pop("plugin")
    b = evolve.Archive.__new__(evolve.Archive)
    b.__setstate__(state)
    assert b.plugin == {}
    assert pickle.loads(pickle.dumps(a)).plugin == {}

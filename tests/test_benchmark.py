"""
test_benchmark.py
-----------------
The device benchmark and --device auto: the CPU is measured, warp is skipped
when there is no CUDA GPU (the warp CPU device is never a training option),
and the cached choice is honoured when it exists. No test needs warp.
"""

import json

from experiments import train

import benchmark  # after train: it puts experiments/ on sys.path


def test_measure_reports_the_cpu(tmp_path):
    result = benchmark.measure(seconds=0.3, workers=1, verbose=False)
    assert result["device"] == "cpu"        # no CUDA GPU on a build machine
    assert result["cpu_eps"] > 0
    assert result["workers"] == 1
    assert result["body_model"] > 0
    path = benchmark.save(result, tmp_path / "device.json")
    assert benchmark.load(path) == result
    assert benchmark.load(tmp_path / "missing.json") is None


def test_auto_reads_the_cached_choice(tmp_path):
    cache = tmp_path / "device.json"
    cache.write_text(json.dumps({"device": "warp", "workers": 1}))
    assert train.resolve_device("auto", None, cache) == ("warp", 1)
    assert train.resolve_device("auto", 3, cache) == ("warp", 3)


def test_auto_measures_when_the_cache_is_missing(tmp_path, monkeypatch):
    cache = tmp_path / "device.json"
    monkeypatch.setattr(benchmark, "measure", lambda *a, **k: {
        "device": "cpu", "workers": 2, "cpu_eps": 9.0})
    assert train.resolve_device("auto", None, cache) == ("cpu", 2)
    assert json.loads(cache.read_text())["cpu_eps"] == 9.0


def test_given_device_never_touches_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "device.json"
    monkeypatch.setattr(benchmark, "load", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(benchmark, "measure", lambda *a, **k: 1 / 0)
    assert train.resolve_device("cpu", 2, cache) == ("cpu", 2)
    assert not cache.exists()

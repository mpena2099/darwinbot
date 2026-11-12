"""tools/speed/analyse_trace.py on a tiny real trace."""

import importlib.util
import os

from darwinbot import evolve

_spec = importlib.util.spec_from_file_location(
    "analyse_trace", os.path.join(os.path.dirname(__file__), "..", "tools", "speed", "analyse_trace.py"))
analyse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyse)


def test_analyse_trace_reports_every_section(tmp_path, monkeypatch):
    tdir = tmp_path / "trace"
    monkeypatch.setenv("DARWINBOT_TRACE", str(tdir))
    evolve.run(n_init=2, n_batches=1, batch_size=3, budget_init=8, budget_child=8,
               workers=1, seed=5, out_dir=str(tmp_path / "out"), variants=2, task="steer")
    text = analyse.report(str(tdir))
    for s in ("Acceptance and compute", "L1", "L6", "L7", "L2 potential"):
        assert s in text
    assert "first variant" in text and "alternating variants" in text
    assert "straight command only" in text

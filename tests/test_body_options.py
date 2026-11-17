"""Body options: parse, round-trip, and constrain the bodies evolution makes."""

import numpy as np
import pytest

from darwinbot import body_options, evolve
from darwinbot import morphology as mo


@pytest.fixture(autouse=True)
def reset():
    body_options.set_active(None)
    yield
    body_options.set_active(None)


OPTIONS = body_options.BodyOptions


def test_defaults_are_todays_limits():
    o = OPTIONS()
    assert (o.allow_wheels, o.allow_passive, o.allow_segments) == (True, True, True)
    assert o.max_servos == mo.MAX_SERVOS
    assert (o.min_legs, o.max_legs, o.max_size_mm) == (0, None, None)
    assert o.bed_mm == 220.0
    assert o.is_default() and body_options.parse("") == o
    assert body_options.parse("").to_string() == ""
    assert evolve.Archive(terrains="flat").body == ""


def test_each_token_and_round_trip():
    cases = {
        "no-wheels": dict(allow_wheels=False),
        "no-passive": dict(allow_passive=False),
        "no-segments": dict(allow_segments=False),
        "max-servos=8": dict(max_servos=8),
        "legs=4..6": dict(min_legs=4, max_legs=6),
        "legs=3": dict(min_legs=3, max_legs=3),
        "max-size=250.5": dict(max_size_mm=250.5),
        "bed=180": dict(bed_mm=180.0),
    }
    for text, fields in cases.items():
        o = body_options.parse(text)
        assert o == OPTIONS(**fields)
        assert body_options.parse(o.to_string()) == o
        assert body_options.parse(" " + text.replace(",", " , ") + " ") == o
    both = body_options.parse("no-wheels,max-servos=8,legs=4..6")
    assert both.to_string() == "no-wheels,max-servos=8,legs=4..6"


def test_bad_tokens_raise():
    for text in ("fins", "no-wheels=1", "max-servos=x", "max-servos=0",
                 "legs=a..b", "legs=5..3", "legs=-1", "max-size=0",
                 "bed=wide", "bed=-4"):
        with pytest.raises(ValueError):
            body_options.parse(text)


def test_set_active():
    assert body_options.active().is_default()
    o = body_options.parse("no-passive")
    body_options.set_active(o)
    assert body_options.active() == o
    body_options.set_active("legs=2..4")
    assert body_options.active() == body_options.parse("legs=2..4")
    body_options.set_active(None)
    assert body_options.active().is_default()


def _respects(m, opts):
    assert m.n_wheels == 0
    assert m.n_servos <= opts.max_servos
    assert opts.min_legs <= m.n_legs <= opts.max_legs


def test_random_bodies_and_mutations_respect_limits():
    opts = body_options.parse("no-wheels,max-servos=6,legs=4..4")
    body_options.set_active(opts)
    rng = np.random.default_rng(4)
    for _ in range(200):
        _respects(mo.decode(mo.random_morphology(rng)), opts)
    for _ in range(200):
        parent = mo.random_morphology(rng)
        _respects(mo.decode(mo.mutate(parent, rng)), opts)


def test_no_options_is_the_old_random_bodies():
    def bodies():
        rng = np.random.default_rng(17)
        return [mo.random_morphology(rng) for _ in range(25)]

    body_options.set_active(OPTIONS())
    with_opts = bodies()
    body_options.set_active(None)
    without = bodies()
    assert all(np.array_equal(a, b) for a, b in zip(with_opts, without))


def test_open_archive_turns_on_its_settings(tmp_path):
    import pickle
    import sys
    sys.path.insert(0, "experiments")
    from replay import open_archive
    from darwinbot import ablate, rules
    a = evolve.Archive(terrains="flat", ablate=(ablate.NAMES[0],),
                       rules=(rules.NAMES[0],), body="no-wheels")
    path = tmp_path / "a.pkl"
    path.write_bytes(pickle.dumps(a))
    try:
        open_archive(str(path))
        assert ablate.active(ablate.NAMES[0])
        assert rules.active(rules.NAMES[0])
        assert not body_options.active().allow_wheels
    finally:
        ablate.set_active(())
        rules.set_active(())
        body_options.set_active(None)

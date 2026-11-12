"""Rewrite tests/golden/golden.json and print what changed.

Legitimate only for an intended behaviour change (physics, score, genome,
export), together with a BODY_MODEL bump or an A/B note saying why. A
refactor that is meant to change nothing must never need it: if the golden
tests fail after one, the refactor changed a result.

    PYTHONPATH=$PWD python tools/update_golden.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import test_golden as g  # noqa: E402


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def main():
    try:
        import build123d  # noqa: F401
        have_export = True
    except ImportError:
        have_export = False
        print("build123d missing: export hashes kept as they are")
    old = json.loads(g.GOLDEN.read_text()) if g.GOLDEN.exists() else {}
    new = {}
    for name in g.CASES:
        sim = g.simulate(name)
        if have_export and sim["buildable"]:
            export = g.export_hashes(name)
        elif have_export:
            export = None
        else:
            export = old.get(name, {}).get("export")
        new[name] = {"sim": sim, "export": export}
    a, b = dict(flatten(old)), dict(flatten(new))
    changed = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    for k in changed:
        print(f"{k}: {a.get(k)} -> {b.get(k)}")
    print(f"{len(changed)} value(s) changed")
    g.GOLDEN.parent.mkdir(exist_ok=True)
    g.GOLDEN.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

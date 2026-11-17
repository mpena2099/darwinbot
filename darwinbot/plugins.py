"""
plugins.py
----------
External tasks and terrains, loaded only when a run names their file.

  python experiments/train.py --plugin my_plugin.py --task backward --terrains bump

A plugin is a Python file that calls `fitness.register_task` and/or
`terrain.register_terrain` at import time (see docs/plugins.md). The archive
stores the file name and the SHA-256 of the bytes it was trained with; every
script that loads the archive asks for the same file and warns when the bytes
changed. Nothing is loaded unless --plugin names it.
"""

import hashlib
import importlib.util
import os
import sys
from typing import Optional

LOADED = []      # paths imported in this process, so pool workers can reload them


def loaded() -> tuple:
    return tuple(LOADED)


def load(path: str) -> dict:
    """Import a plugin file by path; return the record stored in the archive."""
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if path in LOADED:      # already imported: its registrations must not run twice
        return {"name": os.path.basename(path), "sha256": digest}
    name = f"darwinbot_plugin_{len(LOADED)}_{os.path.basename(path)[:-3]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if path not in LOADED:
        LOADED.append(path)
    return {"name": os.path.basename(path), "sha256": digest}


def load_all(paths) -> None:
    """Pool worker initializer: import the plugins the parent loaded."""
    for path in paths:
        load(path)


def require(archive, path: Optional[str] = None) -> None:
    """Load the plugin an archive was trained with. A missing --plugin is an
    error; a different file is only a warning."""
    record = getattr(archive, "plugin", None)
    if not record:
        if path:
            load(path)
        return
    if path is None:
        raise SystemExit(
            f"this archive was trained with plugin {record['name']} "
            f"(sha256 {record['sha256']}); pass --plugin {record['name']} to load it")
    got = load(path)
    if got["sha256"] != record["sha256"]:
        print(f"warning: {path} differs from the plugin the archive was trained "
              f"with ({record['name']}, sha256 {record['sha256']}, now "
              f"{got['sha256']})")

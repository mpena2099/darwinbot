"""
tools/kaggle/islands.py
-----------------------
Run one evolution as several islands on Kaggle CPU sessions at once and merge
them every round. Each island is a copy of train_kernel.py that evolves its own
archive for --batches-per-round batches; when all of them finish, their
archives are merged here (every cell keeps the best elite of any island), the
merged archive is uploaded as a dataset, and the next round starts every
island from it. Four islands give about four times the batches per day.

  python tools/kaggle/islands.py --user me --code-dataset me/darwinbot-code \
      --islands 4 --rounds 3 --batches-per-round 8 --train-args "--preset steer"
  python tools/kaggle/islands.py ... --start checkpoints/steer/archive.pkl   # from an archive

Needs the kaggle CLI logged in, the code dataset (`git archive` of this
repository) and the token dataset described in train_kernel.py. This machine
has to stay online between rounds: it is what merges and restarts the islands.
The merged archive of every round is kept under --out.
"""

import argparse
import glob
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from darwinbot.evolve import merge   # noqa: E402

KERNEL = os.path.join(ROOT, "tools", "kaggle", "train_kernel.py")
SETTLE = 180        # s for the last progress version to finish processing


def kaggle(*args, check=True) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"kaggle {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def set_constant(src: str, name: str, value: str) -> str:
    return re.sub(rf"^{name} = .*$", f"{name} = {json.dumps(value)}", src, count=1,
                  flags=re.M)


def push_island(work, i, a, start_dataset):
    d = os.path.join(work, f"island{i}")
    os.makedirs(d, exist_ok=True)
    src = open(KERNEL).read()
    src = set_constant(src, "TRAIN_ARGS", f"{a.train_args} --batches {a.batches_per_round}")
    src = set_constant(src, "PROGRESS_DATASET", f"{a.user}/{a.name}-island{i}")
    src = set_constant(src, "TOKEN_DATASET", a.token_dataset.split("/")[-1])
    src = set_constant(src, "RESUME_DATASET", start_dataset or "")
    with open(os.path.join(d, "train_kernel.py"), "w") as f:
        f.write(src)
    sources = [a.code_dataset, a.token_dataset] + ([start_dataset] if start_dataset else [])
    meta = {"id": f"{a.user}/{a.name}-island{i}", "title": f"{a.name}-island{i}",
            "code_file": "train_kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": sources, "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f)
    kaggle("kernels", "push", "-p", d)


def wait_kernels(ids, poll):
    left = set(ids)
    while left:
        time.sleep(poll)
        for k in sorted(left):
            status = kaggle("kernels", "status", k, check=False).lower()
            if any(s in status for s in ("complete", "error", "cancel")):
                print(f"  {k}: {status.strip().split()[-1]}", flush=True)
                left.discard(k)


def fetch_archive(dataset, dest) -> str:
    """archive.pkl of a progress dataset (downloading the single file is more
    reliable than the whole dataset)."""
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    for _ in range(5):
        kaggle("datasets", "download", dataset, "-f", "archive.pkl", "-p", dest, check=False)
        for z in glob.glob(os.path.join(dest, "*.zip")):
            with zipfile.ZipFile(z) as f:
                f.extractall(dest)
            os.remove(z)
        path = os.path.join(dest, "archive.pkl")
        if os.path.exists(path):
            return path
        time.sleep(60)
    raise RuntimeError(f"no archive.pkl in {dataset}")


def upload_start(work, a, path, version: bool) -> str:
    """Publish the merged archive as the dataset every island resumes from."""
    dataset = f"{a.user}/{a.name}-start"
    d = os.path.join(work, "start")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shutil.copy(path, os.path.join(d, "archive.pkl"))
    with open(os.path.join(d, "dataset-metadata.json"), "w") as f:
        json.dump({"title": f"{a.name}-start", "id": dataset,
                   "licenses": [{"name": "CC0-1.0"}]}, f)
    if version:
        kaggle("datasets", "version", "-p", d, "-m", "merged islands", "-q")
    else:
        kaggle("datasets", "create", "-p", d, "-q")
    while "ready" not in kaggle("datasets", "status", dataset, check=False).lower():
        time.sleep(30)
    return dataset


def main():
    p = argparse.ArgumentParser(description="darwinbot: islands on Kaggle CPU sessions")
    p.add_argument("--user", required=True, help="Kaggle user name")
    p.add_argument("--code-dataset", required=True, help="user/slug of the code dataset")
    p.add_argument("--token-dataset", default=None,
                   help="user/slug of the token dataset (default: <user>/darwinbot-secret)")
    p.add_argument("--name", default="darwinbot", help="prefix of kernels and datasets")
    p.add_argument("--islands", type=int, default=4)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--batches-per-round", type=int, default=8,
                   help="must fit one 12 h session (about 1 h per batch on 4 cores)")
    p.add_argument("--train-args", default="--preset steer",
                   help="train.py arguments; islands of one run must share them")
    p.add_argument("--start", default=None, help="archive.pkl to start every island from")
    p.add_argument("--out", default="checkpoints/islands")
    p.add_argument("--poll", type=int, default=600, help="s between status checks")
    a = p.parse_args()
    a.token_dataset = a.token_dataset or f"{a.user}/darwinbot-secret"
    work = os.path.join(a.out, "kaggle")
    os.makedirs(work, exist_ok=True)

    start, uploaded = None, False
    if a.start:
        start, uploaded = upload_start(work, a, a.start, False), True
    for r in range(1, a.rounds + 1):
        print(f"round {r}/{a.rounds}: {a.islands} islands"
              + (f" from {start}" if start else " from random bodies"), flush=True)
        for i in range(1, a.islands + 1):
            push_island(work, i, a, start)
        wait_kernels([f"{a.user}/{a.name}-island{i}" for i in range(1, a.islands + 1)], a.poll)
        time.sleep(SETTLE)
        archives = []
        for i in range(1, a.islands + 1):
            try:
                path = fetch_archive(f"{a.user}/{a.name}-island{i}",
                                     os.path.join(work, f"result{i}"))
            except RuntimeError as err:
                print(f"  island {i} left out: {err}", flush=True)
                continue
            with open(path, "rb") as f:
                archives.append(pickle.load(f))
        if not archives:
            raise SystemExit("no island produced an archive")
        merged = merge(archives)
        path = os.path.join(a.out, f"round{r}.pkl")
        with open(path, "wb") as f:
            pickle.dump(merged, f)
        shutil.copy(path, os.path.join(a.out, "archive.pkl"))
        s = merged.stats()
        print(f"round {r}: batch {s['batch']}, cells {s['cells']}, best {s['best']:.2f}, "
              f"QD-score {s['qd_score']:.1f} -> {path}", flush=True)
        if r < a.rounds:
            start, uploaded = upload_start(work, a, path, uploaded), True


if __name__ == "__main__":
    main()

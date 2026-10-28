"""
Kaggle script kernel: train DarwinBot on a CPU session and publish the log and
the archive as a private dataset while it runs.

Kaggle only hands over a kernel's output once the session ends. To follow a run
from outside, this kernel uploads train.log and archive.pkl as a new version of
PROGRESS_DATASET every PUBLISH_EVERY seconds and once more at the end; fetch it
with
    kaggle datasets download <PROGRESS_DATASET> -p progress
    unzip -o progress/*.zip -d progress

Publishing needs an API token. Kaggle hands secrets (Add-ons > Secrets) only to
runs started from its website, not to versions pushed with `kaggle kernels
push`, so the kernel also reads token.txt from a private dataset attached as a
data source (TOKEN_DATASET). Use a token made only for this, so it can be
revoked on its own. Without either the run trains as usual and only skips
publishing.

The code comes from a private dataset holding `git archive` of the repository
(Kaggle may unpack it). Edit TRAIN_ARGS and PROGRESS_DATASET before pushing.

A run longer than one session continues in the next: attach PROGRESS_DATASET
itself as a data source and push again. The kernel copies its archive.pkl to
RESUME_TO under the checkpoints folder, where `train.py --preset` resumes from
it, and the new session publishes newer versions of the same dataset.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import threading

TRAIN_ARGS = "--preset steer --batches 5"
PROGRESS_DATASET = "your-user/darwinbot-progress"
TOKEN_DATASET = "darwinbot-secret"   # slug of the private dataset with token.txt
PUBLISH_EVERY = 900        # s
RESUME_TO = "steer/archive.pkl"   # where --preset steer looks for its archive
TIME_LIMIT = 41000         # s, under the 12 h session limit

WORK = "/kaggle/working"
OUT = f"{WORK}/checkpoints"
LOG = f"{WORK}/train.log"


def fetch_code() -> str:
    """Unpack the code dataset into /tmp/darwinbot."""
    dest = "/tmp/darwinbot"
    os.makedirs(dest, exist_ok=True)
    tars = glob.glob("/kaggle/input/**/darwinbot.tar.gz", recursive=True)
    if tars:
        subprocess.run(["tar", "xzf", tars[0], "-C", dest], check=True)
    else:
        root = os.path.dirname(glob.glob("/kaggle/input/**/pyproject.toml",
                                         recursive=True)[0])
        shutil.copytree(root, dest, dirs_exist_ok=True)
    return dest


def api_token() -> bool:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret(
            "KAGGLE_API_TOKEN")
        return True
    except Exception as e:
        reason = type(e).__name__
    files = glob.glob(f"/kaggle/input/**/{TOKEN_DATASET}/**/token.txt",
                      recursive=True)
    if files:
        os.environ["KAGGLE_API_TOKEN"] = open(files[0]).read().strip()
        return True
    print(f"no KAGGLE_API_TOKEN secret ({reason}) and no {TOKEN_DATASET} "
          "dataset: progress is not published")
    return False


def last_batch() -> str:
    try:
        lines = [l for l in open(LOG) if l.startswith("batch")]
        return lines[-1].strip() if lines else "starting"
    except OSError:
        return "starting"


def publish() -> None:
    """Upload the log and the newest archive as a dataset version."""
    d = "/tmp/progress"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    if os.path.exists(LOG):
        shutil.copy(LOG, d)
    for p in glob.glob(f"{OUT}/**/archive.pkl", recursive=True):
        shutil.copy(p, d)
    with open(f"{d}/dataset-metadata.json", "w") as f:
        json.dump({"title": PROGRESS_DATASET.split("/")[1],
                   "id": PROGRESS_DATASET,
                   "licenses": [{"name": "CC0-1.0"}]}, f)
    msg = last_batch()[:50]
    r = subprocess.run(["kaggle", "datasets", "version", "-p", d, "-m", msg, "-q"],
                       capture_output=True, text=True)
    if r.returncode:        # first upload: the dataset does not exist yet
        r = subprocess.run(["kaggle", "datasets", "create", "-p", d, "-q"],
                           capture_output=True, text=True)
    print(f"[progress] {msg}: {'ok' if r.returncode == 0 else r.stderr[-200:]}")


def main():
    code = fetch_code()
    os.chdir(code)
    slug = PROGRESS_DATASET.split("/")[1]
    previous = glob.glob(f"/kaggle/input/**/{slug}/**/archive.pkl", recursive=True)
    if previous:
        os.makedirs(os.path.dirname(f"{OUT}/{RESUME_TO}"), exist_ok=True)
        shutil.copy(previous[0], f"{OUT}/{RESUME_TO}")
        print(f"resuming from {previous[0]}")
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run(pip + ["mujoco>=3.6.0", "cma>=3.3", "ribs>=0.12"], check=True)
    publishing = api_token()
    if publishing:
        subprocess.run(pip + ["-U", "kaggle"], check=True)
    env = dict(os.environ, PYTHONPATH=code)
    cmd = (f"timeout {TIME_LIMIT} {sys.executable} experiments/train.py {TRAIN_ARGS} "
           f"--workers {os.cpu_count()} --out {OUT} > {LOG} 2>&1")
    run = subprocess.Popen(["bash", "-c", cmd], env=env)
    done = threading.Event()

    def loop():
        while not done.wait(PUBLISH_EVERY):
            publish()
    if publishing:
        threading.Thread(target=loop, daemon=True).start()
    run.wait()
    done.set()
    if publishing:
        publish()
    print(last_batch())


if __name__ == "__main__":
    main()

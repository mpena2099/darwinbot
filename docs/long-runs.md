# Long runs, GPU and Kaggle

## Long runs

A run on the `terrain` preset with 3 workers and the default 2 variants takes
roughly 30 to 40 min for the initial batch and 6 to 8 min for each later batch
(5 s episodes, 0.004 s physics step, 120 CMA-ES evaluations per child; see
`--seconds`, `--variants`, `--budget-init` and `--budget-child`). The archive is saved after
every batch; stopping mid-batch loses only that batch. Keep the machine awake
and the process alive when the terminal closes:

```bash
tmux new -s darwin
systemd-inhibit --what=sleep python experiments/train.py --terrains terrain --batches 40 --workers 3
# detach with Ctrl+b d, reattach with: tmux attach -t darwin
```

## GPU (optional)

The inner CMA-ES loop can run on an NVIDIA GPU through MJX. The CPU remains the
default and works without JAX installed.

```bash
pip install -e ".[gpu]"
python experiments/train.py --device gpu --popsize 128
```

On the GPU, bodies are optimised one at a time and `--workers` is ignored (one
device, and forking with CUDA does not work). `--popsize` defaults to 64 on the
GPU. A strong NVIDIA GPU is needed to beat the CPU: the GeForce 940MX on this
machine is about 12x slower.

## Training on Kaggle

A long run keeps a laptop busy for a day. Two notebooks in `tools/kaggle/` run
on Kaggle's free sessions instead:

- `train_cpu.ipynb`: trains on a 4-core CPU session of up to 12 hours with the
  browser closed. The archive lands in the notebook output; adding that output
  as an input to the next session resumes the run.
- `benchmark_gpu.ipynb`: measures episodes per second on the CPU and on a free
  GPU (MJX) for the same body and batch sizes from 16 to 1024. A GPU only helps
  when it simulates hundreds of copies at once and every new body costs it a
  compile, so this tells whether it pays off here. The same comparison runs
  locally with `python tools/kaggle/bench_gpu.py`. On a free Kaggle GPU (2x T4)
  MJX ran 0.6 to 1.5 episodes per second at any batch size from 16 to 1024,
  against 13 to 15 on the session's 4 CPU cores, spent 17 to 47 s compiling
  every body, and cannot collide a wheel (cylinder) with a box, which rules out
  wheeled robots on steps and rubble. Training stays on the CPU.
- MuJoCo Warp, the newer GPU backend, does much better on the same free GPU
  (one T4 of the pair): stepping the same models with sine targets at 50 Hz
  (`tools/kaggle/bench_warp.py`, physics only, no CPG on either side), it ran
  2 to 3 times the session's 4 CPU cores with 64 parallel worlds, 7 to 9 times
  with 256, 18 to 20 times with 1024 and 22 to 27 times with 4096, for
  quadruped, hexapod and wheeled bodies on flat ground and on rubble, wheels on
  rocks included. A new body costs under 3 s; the first one compiles for about
  a minute. Training with it (`--device warp`) lives on the `warp` branch: it
  trained one of the showcase islands 2.3 times faster than a CPU island
  (see Results). The GPU computes in float32, so a GPU-trained robot is still
  validated on the CPU.

Upload a notebook with *Create > New Notebook > File > Import Notebook*. While
the repository is private, add a Kaggle secret `GITHUB_TOKEN` with read access.

Kaggle hands over a session's output only when it ends. To follow a run while
it trains, push `tools/kaggle/train_kernel.py` as a script kernel instead (with
the code uploaded as a private dataset, see its docstring): every 15 minutes it
publishes `train.log` and `archive.pkl` as a new version of a private dataset,
which `kaggle datasets download <user>/darwinbot-progress -p progress` (then unzip)
fetches. It needs an API token: Kaggle secrets reach only runs started from the
website, so for pushed versions the kernel reads it from a private dataset
(`darwinbot-secret`, a `token.txt` with a token made only for this).

## Islands: several sessions at once

Kaggle runs several CPU sessions at the same time, but one MAP-Elites run is
sequential: every batch picks parents from the archive the previous batch
left. Islands get around it. Each session evolves its own archive for a few
batches, then the archives are merged (every cell keeps the best elite of any
island, which works because the cells are the same in every run) and every
island continues from the merged archive. Four islands give about four times
the batches per day, and evolving apart for a while also tends to help
diversity.

```bash
python tools/kaggle/islands.py --user <you> --code-dataset <you>/darwinbot-code \
    --islands 4 --rounds 3 --batches-per-round 8 --train-args "--preset steer"
```

Each round pushes one copy of `train_kernel.py` per island, waits for all of
them, merges their archives into `checkpoints/islands/round<N>.pkl` (and
`archive.pkl`) and uploads the result as the dataset the next round starts
from; `--start <archive.pkl>` starts the first round from an existing archive.
Keep `--batches-per-round` within one 12-hour session (about one batch per
hour on 4 cores with the `steer` task). The machine running the script has to
be online between rounds, since it merges and restarts the islands.

Archives from any runs of the same setup can also be merged by hand:

```bash
python experiments/merge_archives.py run1/archive.pkl run2/archive.pkl --out merged.pkl
```

## Resuming after a code change

A change to the genome or the physics bumps `BODY_MODEL`, and `train.py --resume`
refuses an archive of an older one. When `evolve.MIGRATIONS` has a path from its
version, the archive can be brought up to date instead of retrained:

```bash
python experiments/migrate_archive.py old/archive.pkl --out migrated.pkl --workers 4
python experiments/train.py --resume migrated.pkl --batches 30
```

Each elite's genes are translated (a change that only touches the physics or the
printed parts leaves them as they are), its body is checked again, and its
controller is searched for `--budget` evaluations (default 60) starting from the
old one, which is kept when nothing better is found. A body that is no longer
buildable is dropped. Cells are recomputed, history and lineage are kept, and
the script prints one table: elites in, dropped, cells out, how many changed
cell, and mean and best fitness before and after. It never overwrites `--out`.
`--seconds S` re-evaluates every elite with episodes of S seconds and stores S in
the result, so a later `train.py --resume` keeps using it.

An archive trained on the default hardware takes the current default profile, so
a change to it (the servo mass, say) applies; a custom kit is kept, with a
warning (`--profile default` replaces it, `--profile keep` keeps it silently).
The migrated archive carries the old run's bias: its elites hold the cells, so
an A/B between two rule versions still needs equal seeds from scratch.

## From an archive to a showcase

Training tunes each controller only long enough to judge its body, so the
robots picked for a showcase go through four more steps:

```bash
python experiments/polish.py --archive merged.pkl --types 13servos-mid-trot,6servos-low-trot --out polished.pkl --workers 8
python experiments/prune.py --archive polished.pkl --types 13servos-mid-trot --out pruned.pkl
python experiments/validate.py --archive pruned.pkl --type 13servos-mid-trot
python tools/check_assembly.py --archive pruned.pkl --type 13servos-mid-trot
```

The whole chain runs with one command. It refuses an existing `--out` and
writes `polished.pkl`, `pruned.pkl`, `validate.txt`, `assembly.txt`,
`showcase.snap.pkl` and a `summary.txt` with each robot's score before and after
polish and prune, its validate verdict and its clash count:

```bash
python experiments/showcase.py --archive merged.pkl --types 13servos-mid-trot,6servos-low-trot --out showcase --workers 8
```

`--skip polish,prune` leaves stages out (the next stage then reads the previous
stage's archive), and `--budget` sets polish's CMA-ES evaluations.

- `polish.py` runs a longer CMA-ES (1200 evaluations, up to three times more
  for bodies with many joints) from each robot's own controller, on the same
  terrains, variants and task. With as many robots as workers, each worker
  polishes one robot; with fewer robots, they go one at a time and each
  generation is scored on all the workers, which gives the same result
  sooner. A 16-servo robot takes about 3.5 hours on one core.
- `prune.py` removes, one at a time, each part whose removal does not lower
  the score, with a short controller search after each cut. A pruned robot
  can change its name (a 7-servo trotter became a 6-servo one).
- `validate.py` passes a robot only when it is robust: mean loss under 30%,
  5th-percentile loss under 60% and fall rate at most 10% on every terrain and
  command, and no terrain where it fails to move at all. `--unseen` also
  checks the grounds the archive never trained on.
- `check_assembly.py` builds the printed parts and reports clashes.

Pruning reruns the score with fresh terrains and variants, so its number is
the one to report.

A snapshot plays a robot recorded on flat ground, walking straight, with any
later version of the code, while an archive needs the code of its BODY_MODEL.
Record one before the genome or the physics changes:

```bash
python experiments/snapshot.py --archive pruned.pkl --types 13servos-mid-trot --out showcase.snap.pkl
python experiments/replay.py --snapshot showcase.snap.pkl --type 13servos-mid-trot
```

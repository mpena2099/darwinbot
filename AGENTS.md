# AGENTS.md

Guidance for coding agents (and people) working on this repository.

## What it is

Evolution of buildable robots: MAP-Elites (pyribs, a CVT archive by behaviour, servo count and cost) over bodies that are trees of modules, CMA-ES over a CPG with two harmonics and load feedback (Tegotae) for each body, simulated in MuJoCo. Bodies use real parts (Feetech STS3215 12 V servos, printed links, ESP32 + 3S LiPo) and any robot in the archive can be exported as STLs, a bill of materials, an assembly guide and a control JSON. A personal hobby and portfolio project. There is no navigation yet. The firmware core exists (portable C++ in `firmware/core`, with a parity test against the Python); only the ESP32 layer is missing, waiting for hardware.

Code, comments, README, the documentation site and this file are in English. `docs/hardware-dados.md` and `docs/formato-controle.md` are in Portuguese. The beginner guide exists in both languages (`docs/guia-evolucao.md` and `docs/evolution-guide.md`): change one, change the other. Guide figures: `tools/make_guide_figures.py`.

## Commands

Run from the repository root with the venv active (`venv/`, Python 3.10).

The documentation lives in `docs/` as a MkDocs Material site (`mkdocs serve` to preview; `pip install -e ".[docs]"`). The README is only the landing page: content changes go to the matching page in `docs/`, not to the README. Private plans in `docs/plano-*.md` stay out of the site (`exclude_docs`) and out of git.

```bash
pip install -e ".[video,cad]"                 # gpu: a separate venv (venv-gpu) with ".[gpu]"
pytest tests -q                               # tests, ~1 min; GPU tests skip themselves without jax
python experiments/train.py --terrains terrain --batches 30 --workers 3
python experiments/train.py --parent-selection uniform   # default: novelty
python experiments/replay.py --list           # robot types in the archive, with duty, height, servos and cost of transport (cot)
python experiments/replay.py --type 12servos-high-trot --speed 0.5
python experiments/validate.py --type <type>  # PASS if it loses < 30% under varied physics
python experiments/export.py --type <type>    # exports/<type>/; --organic adds hollowed links and elbows
python experiments/plot_archive.py
python tools/terrain_sheet.py --archive <archive.pkl>    # docs/img/terrains.png
python tools/make_readme_figures.py --archive <archive.pkl> --types a,b,c,d   # README figures; --types pins the robots shown
python tools/parts_sheet.py exports/<type>    # every printed part on one image
python tools/render_assembled.py --archive <archive.pkl> --type <type>   # the robot built from its printed parts, standing and walking
python tools/check_assembly.py --archive <archive.pkl> --type <type>    # printed parts and servo bodies: clashes and horn holes
python experiments/prune.py --archive <archive.pkl> --types a,b --out pruned.pkl   # cut parts the robot does not need
python experiments/polish.py --archive <archive.pkl> --types a,b --out polished.pkl   # longer controller search for chosen robots
python experiments/merge_archives.py a.pkl b.pkl --out merged.pkl   # islands: best elite per cell
python tools/kaggle/islands.py --user <you> --code-dataset <you>/darwinbot-code   # islands on Kaggle
```

The maintainer's machine is a 4-core laptop: run heavy commands with `nice -n 19`, and use at most 1 worker for tests and short check runs. Long runs go to Kaggle CPU sessions (`tools/kaggle/`).

## Architecture

`darwinbot/`:

- `morphology.py`: body as a tree of modules (`core`, `joint`, `segment`, `link`, `wheel`, `foot` and the passive ones without a servo: `spring`, `tail`, `skid`, `spike`, `caster`), up to `MAX_NODES = 20`, `MAX_SERVOS = 16`, depth 5 (spine segments do not count toward depth, so a limb on the last of six plates is as deep as one on the core). Fixed-size genome, one block of `BLOCK_GENES = 15` genes per node slot, all continuous measures: link length 40 to 200 mm, section 12 to 30 x 3 to 10 mm, elbow (offset 25 to 80 mm, angle 30 to 150°), any roll, mount position and yaw, foot radius 8 to 30 mm, wheel radius 30 to 100 mm and width 10 to 60 mm (the otherwise unused `WIDTH` gene of a wheel block), `PROFILE` and `BEND` (a link, joint or spring straight, L-shaped or arced, bent -90 to +90°). On the core, the unused genes become the contour factors (`CONTOUR_RANGE` 0.4 to 1.4) of the 8 vertices of the torso's star polygon. A foot's `AXIS` gene picks its shape (`FOOT_SHAPES`: sphere, blade, pad, claw). The `segment` module is a spine servo followed by a second torso plate (length and width 80 to 220 mm), attached only to slots 0 and 1 of the core or of another segment, at most `MAX_SEGMENTS = 6`, in a chain through slot 0 of each plate (slot 1 faces the parent). The core has 12 slots: 0 to 7 on the contour and 8 to 11 on the faces (top front, top back, bottom front, bottom back), where the child points up or down. `violations()` applies the printing rules (link bending stress under the servo stall torque <= 25 MPa, parts <= 220 mm); a body that breaks a rule is not buildable and is never simulated. `decode()` accepts any vector and always returns a valid body. The slot is the node's stable id. Mutation works on the tree (a continuous parameter, with a large jump one step in five; stretch or shrink a whole limb; change the type, including into a passive module; add; remove; mirrored duplicate; copy a subtree). Crossover by grafting (`graft`): a whole limb of another elite goes into a free slot, in 20% of the offspring (`GRAFT_P` in `evolve.py`). The training seed is drawn on every run and printed in the log.
- `controller.py`: fixed-size controller genome, indexed by node slot (genes of absent slots are ignored, which lets a child inherit its parent's controller). Per joint: `amp1, phase1, offset, amp2, phase2, sigma, steer_gain, steer_shift`; per wheel: speed and `steer_shift`. One phase oscillator per joint with load feedback (Tegotae): `dtheta = omega - sigma * N * cos(theta)`, `N` = servo load over the maximum torque. The amplitude is clipped by the servo speed. `advance_phases` is compiled with Numba.
- `builder.py`: tree + terrain -> MJCF and the `ActuatorLayout`. The torso is the contour polygon extruded, a segment is a joint plus a box, and an L or arc link becomes 2 or 3 pieces with the accumulated bend. **The actuator order in the layout must match the order in the model** (depth first, children by slot); the controller depends on it. Link mass comes from the geometry (PLA); spring, tail, skid, spike and caster enter as passive joints or geoms with their own friction, with no actuator. Hard plastic (servo housings, links, elbows, torso, tail, caster fork) touches the floor with `PLASTIC_FACTOR` of the terrain friction; only the TPU feet and wheels grip fully. A joint's servo box (`morphology.servo_box`) is fixed to the parent body at the joint frame, since only the horns turn on the robot; it is turned with the joint axis (the axis runs through the 35 mm height) and sits `SERVO_AXIS_OFFSET` (12.5 mm) from the axis along the 45 mm length, as on the STS3215; an elbow plate includes the round boss around the link tip, and `sim.is_buildable` rejects an elbow that cuts into a module it does not carry.
- Printed shapes in the simulation: `builder.py` also collides the parts that have to assemble (massless, next to the servo box or link they belong to): the servo cradle (`cradle{i}`), the round link end bosses, the U fork of a joint's link (`fork{i}_{k}`), the strut and tongue of an elbow that carries a servo, a thin torso plate and a tab under every torso mount. The 60 mm torso prism only meets the terrain (contype 4). `builder.AssemblyRules` says which pairs may overlap (a servo in its cradle, a link in its fork, an elbow on its tip); `sim.is_buildable` rejects any other overlap at spawn, including pairs MuJoCo never collides (parent and child, welded parts), and `builder.joint_ranges` stops each joint before the first clash it causes moving alone (parent and child pairs included); clashes that need several joints at once are left to the episode's self-collision (robot geoms are contype 2 / conaffinity 3 and touch each other). `morphology.fit_rules` and `fit_clashes` repair what `violations` or the spawn check would throw away (they run at the end of `random_morphology`, `mutate` and `graft`, not in `decode`).
- `sim.py`: `Episode` compiles once and runs many controllers; it measures the behaviour descriptors (duty, height); `theta` is simulation state; control runs at 50 Hz (the target is held between updates, as on the ESP32); a body with overlapping parts at the spawn is not buildable.
- `sim_gpu.py`: the same thing batched with MJX. **Any physics or control change in `sim.py` must be mirrored here**; `tests/test_sim_gpu.py` compares the two (it runs only in venv-gpu, ~3 min on the local 940MX).
- Steering by command: two genes per slot (`STEER_GAIN`, `STEER_SHIFT` in `controller.py`); `Params(ctrl, layout, command)` applies the command (-1 left, +1 right) before the speed clip and keeps the values from before it in `*_base`, which is what goes to the JSON (format 3) and the firmware (`set_command`). The `steer` task, the default, runs three episodes per evaluation (`fitness.commands`, `task_score`).
- Tasks: the archive stores `task` (`steer`, `forward` or `turn`, `fitness.TASKS`) and `payload` (extra kg on the torso, passed to `build_mjcf`/`Episode`/`BatchEpisode`); `replay`, `validate` and `export` read both from the archive. `Metrics.yaw` is the accumulated change of the torso heading (computed the same way in `sim.py` and `sim_gpu.py`). The `step` terrain is a 3 cm step 0.50 m from the spawn. Terrains live in `terrain.py` (`TERRAINS`, sets in `PRESETS`); `rough` and `rubble` are random and training draws a new one for every body (`get_terrains(spec, seed)`), while `replay`/`validate` use the fixed one. `Terrain.spawn_z` is the ground height at the spawn (`downhill` starts on a plateau). Genome and descriptors do not change with the task, so `BODY_MODEL` stays.
- `rules.py`: optional score rules (`worse_turn`, `turn_forward`, `tilt`, `slip`, `servo_ground`), off by default, set with `train.py --rules a,b` and stored in `Archive.rules` like `ablate`; `fitness.py` reads them, and the metrics they use (`Metrics.tilt`, `slip`, `servo_ground`) are always measured in `sim.py` and mirrored in `sim_gpu.py`. A rule enters a real run only after its own short run passes (`docs/roadmap.md`).
- `randomize.py`: physics variations (friction, mass, torque, stiffness, backlash, delay) for training and `validate.py`.
- `experiments/lineage.py`: genealogy stored in `Archive.lineage` (the parent of every elite, including replaced ones).
- `inner.py`: CMA-ES per body; `evolve.py`: pyribs CVT archive (`N_CELLS` = 800, five axes; descriptors duty, torso height over reach, number of servos (`MAX_SERVOS`), number of legs (`Morphology.n_legs`) and cost of transport on a log scale, measured on the nominal flat terrain), parent selection by novelty (`--parent-selection novelty|uniform`), type names (`8servos-mid-trot`: servos, height band `low/mid/high`, gait `crawl/walk/trot/belly/roll`), the outer loop, checkpoints. Bodies with more than 4 joints get more CMA-ES evaluations (`scaled_budget`, up to 3x).
- `firmware/core`: the same CPG in dependency-free C++17 (`darwin_control`), which divides the load by `load_full_scale` and wraps the phase every tick; `tests/test_firmware_parity.py` compiles it and compares targets and wheel speeds with `controller.py`.
- `parts.py`, `control_file.py`, `export.py`: build123d parts with each robot's measures, the firmware JSON (format 3, with `load_full_scale` and the steering genes, `docs/formato-controle.md`), and the export package that walks the tree: one STL per distinct size (`link_{L}x{W}x{T}`, with `_L{deg}` or `_arc{deg}` when bent, `elbow_{D}mm_{A}deg_{W}x{T}`, `foot_{R}` (or `foot_{shape}_{R}` for blade, pad and claw, with a socket hub), `segment_plate_{L}x{W}`, `cradle_{axis}` (plus `_w{deg}`, `_back`, `_free`), `wedge_{deg}`, `spring_{W}x{B}`, `tail_{L}`, `skid`, `spike`, `wheel_{R}x{W}`, `caster_wheel_{R}`, `caster_fork`), the torso in bed-sized tiles (`torso_tile_{i}`, or `torso_plate` when it fits) with splice plates (`splice_plate_{span}`), and one assembly line per servo in layout order.

## Planned work

The plan is in `docs/roadmap.md` (Next, Later, and the ROS 2 extension). Notes for implementing it:

- `getup` touches `sim.py` and `sim_gpu.py` (spawn orientation and fall detection) and `fitness.py`.
- Coupled oscillators and the balance sense (IMU) touch `controller.py`, `sim.py`, `sim_gpu.py`, the firmware and `BODY_MODEL`.
- Wider wheels and the six-segment spine already change the genome (`MAX_SEGMENTS = 6`, wheel width in the `WIDTH` gene); `BODY_MODEL` is bumped once, when the branches that change the genome are merged.
- The speed command of the ROS 2 extension is a second gene pair per joint, like the steering one; it also changes the control file format and the firmware.

## Conventions and pitfalls

- The score and the terrains say what a robot must achieve, never how it should look or move: no rewards for symmetry or for familiar shapes. Fix a score rule that can be cheated so it measures its intent; fix the simulator when it allows what the real world does not. Accept a scoring change only after a short run against the current rule shows it does not shrink the archive's variety (cells, body types). Before changing a rule or an archive axis, check whether an earlier decision relied on it.
- An incompatible change to the genome or the physics: bump `BODY_MODEL` in `evolve.py`; old archives are then refused with a clear message.
- Hardware values come from `docs/hardware-dados.md`. The servo measures in `parts.py` come from the SO-ARM100 STEP model of the STS3215, not from a caliper, until someone measures a real servo.
- `.scratch/` is a scratch area outside git; `checkpoints/`, `exports/` and `*.mp4` stay out too.
- Commit messages and pull request descriptions carry no AI attribution.

# Roadmap

## Directions

Where the project could go beyond the items below, in the order that makes
sense: the first two need a built robot and come first; the other three build
on them.

1. **Close the loop with reality**: build a robot, measure how it really
   walks (friction, gear play, part masses, speed), and feed the difference
   back into the simulator, so every robot built makes the next ones more
   faithful. Done with cheap home-built robots, this is a research
   contribution in itself.
2. **Adapt on the real robot**: the built robot tunes its own gait in a few
   minutes of guided trials, covering what the simulator got wrong and even a
   broken leg (see "Adapting on the real robot with Bayesian optimisation"
   under Later).
3. **Evolve from the parts you have**: the user lists a parts inventory (eight
   servos, two wheels, a spool of PLA) and evolution only uses what is there;
   the same kit can be taken apart and rebuilt into another robot for another
   task.
4. **A robot compiler for a task**: the user states the job ("carry 300 g over
   gravel, at most this many servos") and gets a printable robot back: parts,
   bill of materials, assembly guide and firmware, without running a training
   by hand.
5. **A community archive**: people who build a robot share photos, video and
   measurements; the measurements feed direction 1 with data from many robots
   and the project becomes a gallery of real builds (GitHub Discussions, and
   the Hugging Face idea under Later).

## Next: release 3

showcase-2 shipped larger wheels, up to six torso segments, the servo count as
an archive axis, the speed command, hardware kits, robots written by hand in
YAML, plugins and terrain parameters, URDF export and the optional score rules
(worse turn, forward turns, tilt, slip, servo on the ground). Its legged robots
mostly slide on their torso, and training is slow. Release 3 is one run from
scratch (long9) that brings together, each checked first by a short run
against the current rules:

- **Robots that stand instead of sliding**: the torso's share of ground
  contact is now measured (`body_ground`) and a robot that slides on its body
  is named `belly`. To compare: a score rule for it (`--rules body_ground`,
  sliding scrapes the plate and the electronics) and body ground contact as a
  sixth archive axis (`--axes body`), so standing and sliding robots stop
  competing for the same cell.
- **Faster training**: CMA-ES steps focused on the genes a mutation changed,
  scoring candidates on one physics variant and the best on all (2 to 4 times
  fewer inner-loop evaluations in a benchmark), fewer random bodies on mature
  archives, and a race rule that stops a hopeless child early.
- **Cheap evaluation first** (multi-fidelity): a quick screen, such as straight
  on flat ground with a small budget, drops the worst children before the full
  test.
- **Protecting new bodies**: a new body nearly always loses to the old elite of
  its cell because its controller is still poorly tuned, not because the body
  is worse (Cheney et al., 2018). A grace period, or a larger budget after a
  big body mutation, lets an innovation live long enough to show what it can
  do.
- **Mirrored mutation**: a mutation changes one part at a time, so a body
  that started symmetric drifts away from it and nothing brings it back. With
  some probability the same change also goes to the mirror part. The score
  still says nothing about symmetry: a symmetric body only becomes easier to
  reach, and keeps its cell only if it moves better.
- **Coupled oscillators**: each joint's oscillator also feels its neighbours'
  phases, the way the spinal circuits of each leg synchronise a gait, instead
  of coordinating only through the body and the phase genes. Control only, no
  new part.
- **A showcase GIF of four environments**, with the flat and rubble island that
  long8 lacked: the best robot trained on flat ground, on flat ground and
  steps, on flat ground and rubble, and on the full set, each on its own
  terrain.
- **A change of grip mid-stride** (suggested in
  [discussion #50](https://github.com/mpena2099/darwinbot/discussions/50)):
  a strip test next to `validate.py` runs a chosen robot with a short
  low-friction strip at several positions and reports how often it stays
  upright and walks a body length past the far edge. Evaluation only.
- **Repeatable runs** (`--deterministic`, off by default): a batch's results
  are taken in the order they were sent instead of the order they finish, so
  the same seed and worker count repeat a run exactly, for anyone who wants to
  reproduce a result. A little slower, since a batch waits for its slowest
  body.
- **MuJoCo Warp on NVIDIA GPUs** (`--device warp`): the backend that trained
  long8's GPU island, kept on a branch until now, joins the main code with the
  same physics as the CPU loop.
- **MuJoCo 4**, after the run's code is fixed: a separate step, with the
  compiled episode loop reading the official structure offsets and every
  golden and parity test checked again.

## After release 3

Release 4 puts training speed first, then three changes that open body plans
the current limits hold back.

- **Shorter release cycles** (release 4, how the release is run): release 3
  took about a week from its first gate to the end of training. To shorten it:
  - code-only releases (firmware, tools, docs) ship on their own, without
    waiting for a training run from scratch;
  - every gate is sized right from the start, two seeds and enough batches,
    all arms in parallel on Kaggle, instead of short gates that show nothing
    and have to be run again;
  - the speed levers below, worth 2 to 3 times less time per batch;
  - a Kaggle GPU island whenever the weekly quota allows it.
- **Faster training, the levers left out of release 3** (release 4, first):
  release 3 brings the compiled loop's successors (diagonal and focused
  CMA-ES steps, one physics variant per candidate, a quick screen before the
  inner loop), worth about 2 to 3 times fewer evaluations per body. Still to
  build and gate, each against the current search:
  - a race rule that stops a child early when its first evaluations cannot
    beat the elite of its cell;
  - fewer random bodies once the archive is mature, since most random bodies
    are rejected after a full inner loop;
  - fewer evaluations per body on a mature archive, where children differ
    little from their parents;
  - a cheaper `mj_step`: collision geoms that only matter for assembly, and
    solver iterations, measured first and changed only if the physics stays
    the same.
- **The score rules, one at a time** (release 4): `energy_abs`, `turn_forward`,
  `fall_zero` and `body_ground` each passed or nearly passed a short gate,
  but together they left almost no robot that moves (9 and 0 movers against
  35 and 31 without them, two seeds, flat, 22 to 37 batches), so long9 runs
  without them. Each one gets its own run against no rules, from scratch,
  with two seeds and at least 80 batches, and the user's review in replay,
  before it enters a long run. A rule that drops the movers fails.
- **Coupled oscillators on the GPU backends** (release 4): `sim_gpu.py` (MJX)
  and `sim_warp.py` (MuJoCo Warp) do not simulate the coupling that release 3
  adds to the controller, so `--device gpu` and `--device warp` refuse to
  train until they do and a GPU session checks their parity with the CPU.
- **Turning in place** (release 4): the `steer` task trains turns while the
  robot walks at full speed, so on the real robot a turn at speed 0 only bends
  its slowest gait. An extra episode, turning with as little displacement as
  possible, would teach it. It enters the score as one more term, weighted
  like the half-speed episode, so a robot that only spins cannot lead; the
  gate checks that variety does not shrink and that spin-only robots gain no
  cells, and weighs the extra quarter of evaluation cost.
- **Balance sense, optional** (release 4): an IMU on the ESP32 (an MPU-6050, for example)
  feeding the torso's tilt back into the controller through evolved gains that
  start at zero, so each leg can correct a lean. Today the controller only
  feels each servo's load, so evolution favours bodies that are stable on their
  own. A robot built without an IMU keeps working: the gains are simply unused.
- **A fixed passive wheel** (release 4): a small free-rolling wheel whose axle
  is fixed to its segment, so it rolls easily along the body and resists
  sliding sideways, the way a snake's belly scales do and the way snake robots
  such as Hirose's ACM move. Today `caster` swivels freely and `skid` slides
  the same way in every direction, so lateral undulation has nothing to push
  against. Cheap to print and plain to simulate, since MuJoCo already handles
  free wheels.
- **A review of the body limits** (release 4): `MAX_NODES = 20` modules and
  `MAX_SERVOS = 16` servos fit a dog or a two-servo-leg spider, but leave a
  centipede with about five leg pairs once six spine servos are spent. Raising
  them grows the genome and the inner loop's search, and a robot with 24
  servos needs another battery and wiring (16 STS3215 already peak above
  30 A on a 3S pack), so each step up is weighed against training time and
  build cost.
- **Build one robot**: no robot has been built yet. `tools/check_assembly.py`
  and `tools/render_assembled.py` already check the printed parts on screen;
  next, measure a real STS3215, correct `parts.py`, and print one bracket,
  link and horn adapter to test the fit before a whole robot.
- **A search that learns where to mutate** (CMA-ME / CMA-MAE): in these
  MAP-Elites variants a CMA-ES learns which directions in the genome fill new
  cells or improve old ones; pyribs ships them. A first try perturbed all 300
  genes at once and most bodies came out unbuildable, so the emitter has to
  move only the genes of modules that exist, with smaller steps, or repair
  bodies before they are scored.
- **Adversarial terrains**: keep the terrains and physics variants where the
  current elites fail most and score everyone on them too, the way Hillis
  evolved hard test cases as "parasites" and POET evolves terrains alongside
  its walkers. A hall of fame of these hard cases would also make
  `validate.py` stricter.
- **Getting up** (`--task getup`): the robot starts upside down and scores by
  ending upright, so a fall no longer ends the episode.
- **Mutations proposed by a language model**: an LLM reads a body and its
  scores and proposes a change, as one mutation operator among the others. It
  pulls against the idea of evolution finding bodies on its own, so it is an
  experiment: it supplies only part of the offspring and a run with it is
  compared with a run without it.
- **Grounds drawn as images**: terrains already take parameters on the command
  line and plugins add new ones; next, draw the ground as a greyscale PNG
  (`heightmap:yard.png:height=0.05`), stored inside the archive so `replay`
  and `validate` still work on another machine.
- **Your own parts, controllers and sensors**: today a module kind (wheel,
  joint, foot...) is spread over the genome, the simulator, the printed parts
  and the exporter, so adding one means editing the core. Each kind becomes a
  self-contained package (its genes, mounting rules, simulation model, printed
  parts, bill of materials and hardware numbers), loaded with `--modules`
  like terrain and task plugins. The same goes for the controller (the CPG
  becomes one implementation among others) and for sensors. First example: a
  wheel driven by a DC motor.
- **Hardware profiles, the next steps**: every hardware number already comes
  from a kit of one YAML file per component (servo, battery, electronics,
  material, printer), stored in the archive. Still to do: (1) a hardware tab in
  the command builder page that writes the kit next to the `train.py`
  command; (2) servos without a load sensor (common PWM servos), with the load
  feedback gain held at zero; (3) a parametric servo housing, so the cradle,
  fork and elbow parts and `tools/check_assembly.py` work for other servo
  sizes, the largest step; (4) firmware drivers per servo family (Feetech
  serial, PWM, Dynamixel) under the portable core.
- **Archives and exports on Hugging Face** (an idea): publish the showcase archives and the export packages (STLs, bill
  of materials, assembly guide, control file) as a Hugging Face dataset, and
  perhaps a Space that shows the robots walking in the browser.
- **Migration between islands** (an open idea, not a plan): islands train
  apart and only meet when their archives are merged for the showcase. Merging
  them now and then and letting each resume from the merged archive, with its
  own seed, could let a limb found on one island be grafted on a body from
  another. Against it: the islands would grow alike, losing what makes them
  worth having; the showcase merge already keeps the best robot of every
  cell, so the gain is only the new robots cross-island grafts would make,
  which is not shown; a weak island spreads to the others; and every session
  change gets one more step that can fail. `tools/kaggle/islands.py` already
  resumes islands from a merged archive.
- **Walking or rolling as the ground asks** (release 4, a wheel-stall switch,
  no IMU needed): a robot with legs ending in wheels rolls by default and walks
  when it is stuck. The wheels drive while the legs hold a posture; when the
  wheels are commanded to turn but barely move (the gap between commanded and
  measured wheel speed, both reported by the STS3215, with high load) for a
  while, the robot switches to its walking rhythm, and goes back to rolling
  after some time. The stall threshold, the delay and the walking time are
  genes, next to two sets of rhythm genes (roll and walk). Training needs
  courses that mix flat ground and a step or rubble in the same episode, so
  that switching pays. A wheel spinning free on loose ground does not stall and
  is not detected without odometry; stuck against a step or an obstacle is.
  It changes the controller, the firmware and the control file, with a new
  `BODY_MODEL` and its own gate. A first try (branch `tachikoma`) scaled the leg
  amplitude by a low-pass of the wheels' load and did not work: two 8-batch
  runs kept the variety (46 and 52 cells) but scored lower than without it
  (best wheeled-legged 0.35 and 0.25 against 0.46 and 0.45), because a wheel
  rolling freely already carries load (the signal sat between 0.4 and 0.6) and
  evolution used the gene as a fixed leg weakener.
- **A surrogate model**: a small model trained on the thousands of bodies the
  archive has already scored would predict a new body's score before it is
  simulated, so only promising ones are; it complements cheap evaluation
  first.
- **Learned archive axes** (AURORA): today's axes (contact time, height,
  legs, cost of transport) are chosen by hand. Axes learned from recorded
  behaviour could separate gait styles nobody listed. More experimental.
- **Organic parts on export, beyond flat plates**: `export.py --organic`
  hollows links and elbow plates with 2D topology optimisation. Still to do:
  the torso tiles (many mounts, no single clamp), a 3D optimisation for
  out-of-plane loads, and scoring the lighter robot in simulation before
  trusting it.
- **Elastic tendons across joints**: printed TPU tendons that span one or two
  joints and store energy, as a kangaroo bounces on its tendons and a
  hamstring moves hip and knee together (MuJoCo simulates tendons natively).
- **Direction-dependent friction**: printed scales that slide forwards and
  grip backwards, the way a snake's belly scales push it along.
- **Snakes and centipedes**: the spine already holds up to six segments, but
  lateral undulation also needs the direction-dependent friction above;
  sidewinding and concertina motion work without it.
- **Growth rules for the body**: instead of describing a body part by part,
  the genome would hold a rule that grows it ("a segment sprouts a leg on each
  side and another segment, five times"), as L-systems and Karl Sims'
  recursive graphs do and as genes do in animals. Symmetry and repetition come
  for free, so a centipede or a snake is one short rule rather than twenty
  tuned nodes. It replaces the genome, so it needs a new `BODY_MODEL`.
- **Controllers as evolved programs** (an idea only): genetic programming
  could invent the controller's rules, such as reflexes driven by servo load,
  instead of tuning the fixed CPG. Language is not the obstacle (the ESP32 runs
  MicroPython, and a small expression tree is cheap at 50 Hz); the work is
  making the firmware run any evolved program safely (bounded size and time,
  no division by zero or runaway values) and testing parity with the Python
  for arbitrary programs rather than one formula.
- **Closed loops grown by evolution**: a "ligament" gene that pins the tip of
  one part to another part of the body, so linkages such as four-bar
  mechanisms can arise on their own (a wrasse's jaw and the crossed ligaments of
  a knee are four-bar linkages) instead of being handed over as a ready-made
  module. Hard parts: the loop must close at the spawn pose to be buildable,
  closed chains are less stable in simulation, and printing needs pins and play.
- **Shapes drawn by an evolved network** (layer 4): a small CPPN in the genome
  draws each foot and the torso as an implicit surface (as Cheney et al. did for
  soft robots in 2013), split into convex pieces for MuJoCo and checked for
  strength by finite elements.
- **Adapting on the real robot with Bayesian optimisation**: once a robot is
  built, use the archive as a prior and try a handful of its gaits on the real
  robot (about ten trials), as in Cully et al.'s Intelligent Trial and Error
  (2015), until one works on the real floor; the same method lets a damaged
  robot find a new gait. Bayesian optimisation can also tune the simulator
  (friction, gear play, delay) to match servo measurements on the bench.
- **Formulas that explain the archive** (symbolic regression): search for
  short equations behind the results, such as how speed grows with wheel
  radius or leg length, or condense an evolved controller into a readable
  rule; on the bench, fit the real servo's friction from measurements. An
  analysis tool for the Results page and the simulator, not a source of
  robots.
- **A walk or roll command**: a second command next to steering, with two genes
  per joint, trained by testing each robot in both modes (walking on steps,
  rolling on flat ground). Predictable, and fits the ROS 2 extension, which
  would send "roll" or "walk" as a high-level order; costs more episodes per
  evaluation.
- **Foot contact sensors**: a microswitch or force sensor (FSR) in each foot,
  like the touch receptors of a sole. Today the robot feels the ground only
  through each servo's load, an indirect and noisy signal; the load feedback
  could use real contact instead. Cheap and light, one wire per foot.
- **A mini robot on hobby micro servos** (an idea, a separate line): an
  MG90S costs about a tenth of an STS3215 and weighs 13 g, but gives about
  0.2 N*m (15 times less), takes one PWM wire per servo (16 need a PCA9685
  board), and reports neither its position nor its load, so the load feedback
  of today's controller (Tegotae) would go or need extra sensors. Bodies,
  printed parts and the servo model in the simulation would all shrink and
  change, with a training of its own; robots evolved for the STS3215 would not
  carry over. A cheap way to build a first robot.
- **Servos of different sizes**: Feetech makes smaller servos on the same bus
  (the STS3032, lighter and weaker). Evolution would choose, joint by joint,
  between a strong heavy servo and a weak light one, like the large hip muscles
  and the small ones of the toes.
- **Distance sensing**: a time-of-flight sensor (a VL53L0X, for example) as a
  simple eye that measures how far the obstacle ahead is, and later a camera
  (the ESP32-S3 takes an OV2640). A small 2D time-of-flight lidar on the
  ESP32's serial port goes one step further: evolution would see only a few
  channels of it (free distance ahead, left and right, feeding the steering
  command), while the full scan serves ROS 2 (below). Needed for avoiding
  obstacles, navigating and predator and prey, so it comes with a task that has
  a goal in the world.
- **Predator and prey**, much later: a pursuer to escape or a target to catch,
  first scripted, then evolving in a second archive. It needs the robot to
  sense the other one, so it waits for sensors.

## Extension: missions with ROS 2

Beyond the project's own goal. DarwinBot evolves a body and a way of moving,
roughly what a spinal cord does for an animal; a mission (explore, find
something, come back) needs a brain on top that decides where to go and why.
ROS 2 already has that layer (Nav2 for routes and obstacles, perception,
task state machines), so the idea is to plug a finished robot into it rather
than evolve missions. What it would take:

- **Already in place**: `export.py` writes each robot as URDF next to the STLs,
  so it opens in RViz, Gazebo and MoveIt, and the speed command (a second gene
  pair per joint) lets a robot go slower or faster when told, the forward part
  of a Nav2 `Twist`.
- **Odometry**: navigating needs the robot to estimate where it is. Counting
  steps drifts badly on legs, so this needs the IMU and, better, the distance
  sensor or a camera (see Later).
- **micro-ROS on the ESP32**: an optional firmware layer around the existing
  control core that publishes `joint_states` (position, speed and load of
  every servo, and the IMU once there is one) and takes the `Twist`, mapped to
  the commands the core already takes. The robot does not need ROS 2 to walk.
  This is also the shape of the missing ESP32 layer: the gait stays on the
  robot, the brain runs on a PC over Wi-Fi, as in the small micro-ROS robot
  kits sold for teaching (ESP32, 6-axis IMU, 2D lidar, camera).
- **Mapping**: with the 2D lidar publishing `/scan`, the usual ROS 2 SLAM and
  Nav2 stack builds a map and plans routes; the evolved gait follows the
  `Twist` it sends.
- **Sideways walking**: a third command, lateral speed, would let a robot
  strafe without turning its body (the `linear.y` of a `Twist`), useful to
  dodge or line up with a door. Today the score measures progress along the
  torso's axis and charges sideways drift, so a crab-shaped robot can appear
  but must walk along its own axis. It needs a new gene pair per joint, a
  firmware change and a task that scores lateral motion, with its own gate.
  Only after turning works well.

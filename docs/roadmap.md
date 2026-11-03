# Roadmap

## Next

- **Polish the controllers after training**: a controller search ten times
  longer (3600 evaluations from the elite's own controller) raised the same
  bodies by 45 to 75% (walker 0.17 to 0.30, trotter 0.29 to 0.43, roller 0.60
  to 0.86), so elites enter the archive far from what their body allows.
  Spending that search inside training does not pay: with a quarter of the
  children keeping their parent's body and searching three times longer, two
  short runs filled 37 and 42 cells against 44 and 51, for similar best scores.
  The search belongs after training, on the robots picked for the showcase;
  `experiments/polish.py` runs that pass from each chosen elite's own
  controller and writes the polished robots to a new archive.

- **Assemble every export virtually, then build one**: no robot has been
  assembled yet, even on screen. A script that places every exported part and
  a servo model in the spawn pose would check that holes and screws line up,
  that no two printed parts intersect, and that the printed parts match the
  shapes the simulator used, and write the assembled robot as one 3D file to
  inspect. In parallel, measure a real STS3215, correct `parts.py`, and print
  one bracket, link and horn adapter to test the fit before a whole robot.
- **A search that learns where to mutate** (CMA-ME / CMA-MAE): bodies come
  from random mutations today. In these MAP-Elites variants a CMA-ES learns
  which directions in the genome fill new cells or improve old ones; the body
  genome is already a fixed-size continuous vector, the case they are made
  for, and pyribs ships them. Measured against the current search with a short
  run.
  A first try perturbed all 300 genes at once and most emitted bodies came out
  unbuildable, wasting much of each batch; the emitter has to move only the
  genes of modules that exist, with smaller steps, or repair bodies before
  they are scored.
- **Cheap evaluation first** (multi-fidelity): every child pays the full
  evaluation (four terrains, three commands, two physics variants) even when
  it is hopeless. A quick screen first, such as straight on flat ground with a
  small CMA-ES budget, would drop the worst before the full test and could
  make training several times cheaper.
- **Protecting new bodies**: a new body nearly always loses to the old
  elite of its cell because its controller is still poorly tuned, not because
  the body is worse (Cheney et al., 2018). A grace period, or a larger budget
  after a big body mutation, lets an innovation live long enough to show what
  it can do.
- **Score the worse of the two turns**: the `steer` task averages the left and
  right turns, so a robot that only turns one way, such as a one-servo wheel
  dragging itself in circles, still earns half the turn credit. Counting the
  worse turn instead rewards only robots that obey both commands. To be
  measured with a short run against the current rule.
  A first single-seed short run was mixed: straighter robots and more legged
  gaits, but the top ten turned both ways less well than under the current
  rule. Needs three seeds before a decision.
- **Turn forwards, not backwards**: a turn earns its heading change up to the
  distance the torso travelled, in any direction, so a curve driven in reverse
  still scores. In the 92-cell archive of the long3 run, 14 of the 114 turning
  episodes that earned credit drove the curve in reverse, and only 1 of the 18
  among the ten best robots. Counting only forward progress in a turn would
  close the gap; one short run against the current rule measures what it costs.
- **Servo count back as an archive axis**: nothing in the score or the
  archive favours a simpler body, and the best robots use 13 to 16 servos, the
  most expensive to build. With the servo count as an axis next to the number
  of legs, simple robots get cells of their own instead of competing with the
  big ones; the score inside each cell stays the same. It changes the archive
  cells, so it needs a new run.
- **Torso tilt**: only a full fall costs score today, so a robot that runs
  tilted, close to tipping over, loses nothing. A term for the torso's mean
  tilt, or distance multiplied by a stability factor as in some quadruped
  studies, would favour bodies that carry themselves upright.
- **Feet that slip**: a foot sliding while it bears weight leans on the
  simulator's friction model and tends to fail on a real floor. Legged robot
  training commonly subtracts the horizontal speed of feet in contact.
- **A minimum distance before a fall**: falling forward is a classic exploit
  (Karl Sims' creatures learned to topple instead of walk). None of today's
  top robots fall, but scoring zero for a fall before a minimum distance
  closes the door cheaply.
- **Servo bodies touching the ground**: some robots lean on a knee, the servo
  housing, instead of the foot. That is not wrong in itself (gorillas walk on
  their knuckles), so it should not be banned; but the housing was not made to
  drag on the floor and its simulated friction is a guess. Give servo housings
  a measured friction and a wear cost, or let the export add a printed cap
  where a robot uses them, so the choice stays evolution's and stays honest.
- **Adversarial terrains**: instead of drawing rough and rubble terrains at
  random, keep the terrains and physics variants where the current elites fail
  most and score everyone on them too, the way Hillis evolved hard test cases
  as "parasites" and POET evolves terrains alongside its walkers. A hall of
  fame of these hard cases would also make `validate.py` stricter.
- **Getting up** (`--task getup`): the robot starts upside down and scores by
  ending upright, so a fall no longer ends the episode. Bodies that can right
  themselves look very different from bodies that only walk.
- **Mutations proposed by a language model**: an LLM reads a body and its
  scores and proposes a change, as one mutation operator among the others.
  A language model brings what it learned from robots people designed, which
  pulls against the idea of evolution finding bodies on its own, so this is an
  experiment: it supplies only part of the offspring, the usual mutation stays,
  and a run with it is compared with a run without it.

- **Your own environments without code**, in three steps: (1) tune the
  existing terrains from the command line (`--terrains
  "rough:height=0.06,slope:degrees=12,slippery:friction=0.1"`); (2) draw the
  ground as a greyscale PNG (`heightmap:yard.png:height=0.05`, a MuJoCo
  height field like `rough`), stored inside the archive so `replay` and
  `validate` still work on another machine; (3) a complete worked example in
  the Environments page for people who write their own terrain function.

## Later

- **Walking or rolling as the ground asks** (a wheel-stall reflex): a robot
  with legs ending in wheels has one rhythm, legs and wheels together. A gene
  per joint that scales the leg amplitude by a low-pass of the wheels' load
  (branch `tachikoma`) did not work: two 8-batch runs kept the variety (46 and
  52 cells, 19 moving robots each) but scored lower than without it (best
  wheeled-legged 0.35 and 0.25 against 0.46 and 0.45), and freezing the signal
  at its episode mean changed the scores little, so evolution used the gene as
  a fixed leg weakener. A wheel that rolls freely already carries load, and the
  signal sat between 0.4 and 0.6 most of the time. A stall signal from the gap
  between the commanded and the measured wheel speed (the STS3215 reports both)
  would separate free from stalled wheels; it changes the controller, the
  firmware and the control file, with a new `BODY_MODEL`.
- **Servo count as an archive axis**: a 6-servo body and a 16-servo one that
  move alike share a cell today, and the one with more motors usually wins it;
  the ten best robots average 13.8 servos. With the servo count as an axis,
  simple and cheap bodies would keep their own cells next to the complex
  ones. The cells change, so earlier archives no longer fit and training
  starts over.
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
- **More vertebrae**: allow six to eight torso segments instead of two, so
  long bodies with repeated vertebrae can appear, the way nature builds snakes
  and centipedes. Today a body has at most three articulated torso pieces and
  limbs five modules deep, enough for a legless "lizard" with a waving tail but
  not for a snake. Lateral undulation also needs the direction-dependent
  friction above; sidewinding and concertina motion work without it.
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
- **Wider wheel sizes**: wheel radius is limited to 30 to 60 mm, a
  conservative choice rather than a physical limit. Opening it to about 25 to
  100 mm would leave only the real checks: the wheel fits the print bed, the
  servo body clears the ground, and the torque is enough, which the simulation
  already tests. Large wheels could roll over rocks that stop today's ones.
- **Coupled oscillators**: each joint's oscillator also feels its neighbours'
  phases, the way the spinal circuits of each leg synchronise a gait, instead
  of coordinating only through the body and the phase genes. Control only, no
  new part.
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
- **Balance sense**: an IMU on the ESP32 (an MPU-6050, for example) feeding the
  torso's tilt back into the controller, so each leg can correct a lean. Today
  the controller only feels each servo's load, so evolution favours bodies that
  are stable on their own (wide, low, many supports or wheels); a body that
  stands upright on two legs needs this to survive.
- **Foot contact sensors**: a microswitch or force sensor (FSR) in each foot,
  like the touch receptors of a sole. Today the robot feels the ground only
  through each servo's load, an indirect and noisy signal; the load feedback
  could use real contact instead. Cheap and light, one wire per foot.
- **Servos of different sizes**: Feetech makes smaller servos on the same bus
  (the STS3032, lighter and weaker). Evolution would choose, joint by joint,
  between a strong heavy servo and a weak light one, like the large hip muscles
  and the small ones of the toes.
- **Distance sensing**: a time-of-flight sensor (a VL53L0X, for example) as a
  simple eye that measures how far the obstacle ahead is, and later a camera
  (the ESP32-S3 takes an OV2640). Needed for avoiding obstacles, navigating and
  predator and prey, so it comes with a task that has a goal in the world.
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

- **URDF export**: write each evolved robot as URDF next to the STLs (every
  module a `<link>` with its mesh, every servo a `<joint>`), so it opens in
  RViz, Gazebo, MoveIt and the rest of the ROS 2 tools. The body is already a
  tree of links and joints, so this is a direct translation.
- **Speed command**: Nav2 sends a `Twist` with forward and turning speed;
  today a robot takes only the turn command and walks at its own pace. A
  second gene pair per joint, like the steering one, would let evolution learn
  to go faster or slower when told. This is the one part that touches
  evolution itself.
- **Odometry**: navigating needs the robot to estimate where it is. Counting
  steps drifts badly on legs, so this needs the IMU and, better, the distance
  sensor or a camera (see Later).
- **micro-ROS on the ESP32**: an optional firmware layer around the existing
  control core that publishes `joint_states` (position, speed and load of
  every servo, and the IMU once there is one) and takes the `Twist`, mapped to
  the commands the core already takes. The robot does not need ROS 2 to walk.

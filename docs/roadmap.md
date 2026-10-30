# Roadmap

## Next

- **Score the worse of the two turns**: the `steer` task averages the left and
  right turns, so a robot that only turns one way, such as a one-servo wheel
  dragging itself in circles, still earns half the turn credit. Counting the
  worse turn instead rewards only robots that obey both commands. To be
  measured with a short run against the current rule.
- **Hold the heading when told to go straight**: the straight episode lasts
  8 s, so a robot that slowly curves loses almost nothing; over 30 s some of
  today's best rovers end up going sideways or back to the start. Subtract the
  heading change in the straight episode, as the lateral drift already is, and
  count only forward progress in a turn, so a curve driven in reverse earns
  nothing. Both are about obeying the command; one short run measures them.
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

- **Organic parts on export**, an optional step for whoever prints a robot:
  topology optimisation of every printed part after evolution, under the servo stall torque and the loads seen in
  simulation, removing material where nothing is carried. The parts come out
  bone-like and lighter while the evolved body and its interfaces stay the same.
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

# Roadmap

## Next

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
- **Predator and prey**, much later: a pursuer to escape or a target to catch,
  first scripted, then evolving in a second archive. It needs the robot to
  sense the other one, so it waits for sensors.

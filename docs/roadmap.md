# Roadmap

- **Organic parts on export**, an optional step for whoever prints a robot:
  topology optimisation of every printed part after evolution, under the servo stall torque and the loads seen in
  simulation, removing material where nothing is carried. The parts come out
  bone-like and lighter while the evolved body and its interfaces stay the same.
- **Elastic tendons across joints**: printed TPU tendons that span one or two
  joints and store energy, as a kangaroo bounces on its tendons and a
  hamstring moves hip and knee together (MuJoCo simulates tendons natively).
- **Direction-dependent friction**: printed scales that slide forwards and
  grip backwards, the way a snake's belly scales push it along.
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
- **Getting up** (`--task getup`): the robot starts upside down and scores by
  ending upright, so a fall no longer ends the episode. Bodies that can right
  themselves look very different from bodies that only walk.

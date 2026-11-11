# Related work

DarwinBot is not the first project to evolve robot bodies and controllers
together, nor the first to use MAP-Elites for it. What it adds is the whole
path for one hobby parts kit: evolution constrained by real servos, printing
and assembly rules, then STL parts, a bill of materials, an assembly guide and
a firmware control file for any robot in the archive.

- **MAP-Elites for modular robots.** Nordmoen, Veenstra, Ellefsen and Glette,
  [MAP-Elites Enables Powerful Stepping Stones and Diversity for Modular
  Robotics](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.639173/full),
  Frontiers in Robotics and AI, 2021. MAP-Elites over modular robot bodies keeps
  more diverse morphologies than a single-objective search and finds better
  robots through those stepping stones; the same reason DarwinBot keeps an
  archive instead of one winner.
- **Revolve2.** [Revolve2](https://ci-group.github.io/revolve2/), from the
  Computational Intelligence group at VU Amsterdam: a framework for evolving
  modular robot bodies and brains in simulation, with support for building
  and running them physically.
- **Hardware limits on evolvable robots.** Angus, Buchanan, Le Goff, Hart,
  Eiben, De Carlo, Winfield, Hale, Woolley, Timmis and Tyrrell,
  [Practical hardware for evolvable robots](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1206055/full),
  Frontiers in Robotics and AI, 2023. From the Autonomous Robot Evolution
  project: manufacturing, assembly and power delivery rule out many simulated
  designs, and filtering or repairing them reduces the population's diversity.
  DarwinBot meets the same trade-off with its printing rules, assembly checks
  and repairs (`morphology.fit_rules`, `fit_clashes`).

Where DarwinBot differs:

- Bodies are continuous trees (any link length, bend, mount position and
  torso outline) rather than a grid of fixed modules, and every one is checked
  against printable parts and servo housings before it is simulated.
- Every robot in the archive exports to printable parts and a control file for
  the same CPG, whose C++ core is tested against the Python one.
- It has not been built yet: the step from simulation to a real robot, and
  feeding what a real one measures back into the simulator, is the
  [roadmap](roadmap.md)'s "Assemble every export virtually, then build one".

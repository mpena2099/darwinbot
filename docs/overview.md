# Overview

Two nested loops:

1. **Outer loop, MAP-Elites over behaviour.** The archive is a pyribs
   CVTArchive with about 500 cells over four descriptors measured in the nominal
   flat-floor episode: `duty` (fraction of time the ground-touching tips are on
   the floor), `height` (mean torso height over the body reach), `legs`
   (limbs on the torso with a joint and no wheel, so a hexapod never competes
   with a two-legged crawler for a cell) and `cot`, the cost of transport (mechanical work per
   weight and distance, on a log scale). The last one lets an efficient walker
   keep its own cell next to a faster but hungrier roller. Each cell keeps the best body found there, so a run ends
   with a map of specialists instead of a single design. Parents are drawn with
   probability proportional to their novelty (mean distance to the 10 nearest
   elites in descriptor space), which pushes the search towards sparse regions.
2. **Inner loop, CMA-ES over the controller.** For a fixed body, CMA-ES searches
   the controller genes that this body actually uses. A promising body is
   therefore judged by the best gait it can learn, not by a lucky random
   controller.

Children are mutated copies of elites, or grafts of two elites. The child starts its inner loop from the
parent controller, so learned gaits are inherited along with the body
(Lamarckian inheritance). Controller genes are indexed by the stable slot id of
each module, so a mutation that keeps a module keeps its control. Only the genes
that affect the child are optimised; the rest stay in the genome and become
active again if a module returns.

Construction rules act as natural selection by the printer. A body is buildable
only if every part fits a 220 mm print bed, every link bends under the 2.94 N*m
servo stall torque with a stress of at most 25 MPa (half of what PLA takes), and
no parts overlap at spawn. The torso is not limited: its outline is split into
bed-sized tiles joined by splice plates, and each spine segment is its own plate. A body that breaks a rule scores as unbuildable
and is never simulated.

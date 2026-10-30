# Body genome

All genes are normalised to `[0, 1]` and mapped linearly onto their physical
range on decode, so one mutation sigma works for every gene.

## Module tree (`darwinbot/morphology.py`)

A tree of modules: `core`, `joint`, `segment`, `link`, `spring`, `wheel`, `foot`,
`tail`, `skid`, `spike` and `caster`. Joints, segments (a spine servo followed by
a second torso plate, at most `MAX_SEGMENTS = 2`, attached only to the front or
back slot of the core or of another segment) and wheels carry an STS3215 servo;
links, springs, tails, skids, spikes, casters and feet are passive. The genome is
a fixed vector of `MORPH_LENGTH = MAX_NODES * 15 = 300` genes: one block of 15
genes per node slot, with `MAX_NODES = 20`. Block 0 is always the core; the block index
is the stable slot id of the node. A block is

| Gene | Meaning |
|------|---------|
| `active` | the module exists when > 0.5 |
| `parent` | slot id of the parent module |
| `slot` | attachment slot on the parent (core: 12 slots, 8 on the contour (front, back, three on each side) and 4 on the flat faces (top front, top rear, bottom front, bottom rear); segment: 8 slots like the core; joint, link and spring: tip and two sides) |
| `rot` | roll of the attachment around the slot direction, any angle in [0, 2*pi) |
| `type` | joint, link, spring, wheel, foot, tail, skid, spike, caster or segment |
| `axis` | joint turns about pitch, yaw or roll; for a foot, the shape (sphere, blade, pad or claw) |
| `length` | link length 40 - 200 mm (tail: bar length) |
| `wheel` | wheel or caster radius (wheel 30 - 60 mm, caster 15 - 35 mm), or foot radius 8 - 30 mm; for a tail, the counterweight 0 - 100 g |
| `flip` | joint axis reversed (how a mirrored copy keeps the same signal) |
| `width` | link section width 12 - 30 mm |
| `thick` | link section thickness 3 - 10 mm; for a spring, the TPU blade 2 - 8 mm |
| `offset` | on the core: mount position along the torso edge (on a top or bottom face, across the torso); on a link side slot: elbow distance 25 - 80 mm |
| `angle` | on the core: mount yaw up to +/-60 degrees (on a face, yaw about the vertical, +/-180); on a link side slot: elbow angle 30 - 150 degrees |
| `profile` | link, joint or spring path: straight, L (two pieces) or arc (three pieces); ignored by other modules |
| `bend` | signed bend of an L or arc profile, -90 to +90 degrees |

Passive module physics: a spring is a hinge whose stiffness follows from the TPU
blade thickness (see "Mola de TPU" in [hardware-dados.md](hardware-dados.md)); a tail is a bar
with a counterweight at the tip; a skid has low friction (0.15), a spike high
friction (1.6); a caster is a free wheel on a passive hinge. None has a servo.

For the core block, `length` and `wheel` hold the torso length (0.15 - 0.40 m)
and width (0.10 - 0.25 m), and the unused genes hold the contour factors (0.4 to
1.4) of the 8 vertices of the star-shaped torso polygon, one per contour slot.
For a segment, `length` and `width` set the plate (80 - 220 mm each). `decode()` accepts any vector and always returns a
body within `MAX_NODES`, `MAX_SERVOS = 16` and depth 5: blocks that are
inactive, unreachable, clash on a slot or break a limit are dropped. A joint or
link with no child ends in an implicit foot.

Mutation works on the decoded tree, one to three operators per child: nudge a continuous measure (one step in five
is a large jump; this includes the torso contour, link profile and bend, and foot shape), stretch or shrink a whole limb, change a module type (including
into a passive kind), add a module or small subtree, remove a subtree, duplicate
a subtree mirrored on the opposite slot, or copy it to another free slot. One
child in five is instead a graft: a whole limb of a second elite is attached to
a free slot of the parent, mirrored when the opposite torso side is free, the
crossover Sims (1994) used for his creatures. One body in ten in every batch
is drawn from scratch rather than bred (`random_frac` in `evolve.run`). The initial population mixes
small random trees with seeds that reproduce mirrored leg pairs, with and
without wheels.

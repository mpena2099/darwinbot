# Real parts

Every simulated robot is built from parts you can buy or print: Feetech
STS3215 servos, 12 V version (C018: 2.94 N*m stall, 4.71 rad/s, run at 70% of the torque),
60 g per servo, an ESP32 + 3S LiPo + regulator payload of 0.17 kg in the torso,
printed links of any length from 40 to 200 mm with a width of 12 to 30 mm and a
thickness of 3 to 10 mm (straight, L or arc), a torso plate of free contour with
optional spine plates, blade, pad and claw feet with a socket hub, printed elbow plates that carry the modules mounted on
the side of a link, tails, skids, spikes and caster wheels,
and self-collision between limbs. Link mass comes from the PLA geometry, and a
link must keep its bending stress under the servo stall torque at or below 25 MPa. The
sources and the values still to be checked are in [hardware-dados.md](hardware-dados.md).

The same STS3215 works in two modes, set by the firmware at power-up: position
mode, which turns to an angle within one full turn (4096 steps), for joints,
and wheel mode, which spins continuously at a set speed, for wheels. Evolution
decides each servo's role through the module it carries (`joint` or `wheel`),
and the role stays fixed while the robot runs. In both modes the servo reports
position, speed, load, voltage and temperature; the load is what the
controller's feedback uses.

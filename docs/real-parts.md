# Real parts

Every simulated robot is built from parts you can buy or print: Feetech
STS3215 servos at 7.4 V (1.91 N*m stall, 5.45 rad/s, run at 70% of the torque),
60 g per servo, an ESP32 + 2S LiPo + regulator payload of 0.13 kg in the torso,
printed links of any length from 40 to 200 mm with a width of 12 to 30 mm and a
thickness of 3 to 10 mm (straight, L or arc), a torso plate of free contour with
optional spine plates, blade, pad and claw feet with a socket hub, printed elbow plates that carry the modules mounted on
the side of a link, TPU flexure springs, tails, skids, spikes and caster wheels,
and self-collision between limbs. Link mass comes from the PLA geometry, and a
link must keep its bending stress under the servo stall torque at or below 25 MPa. The
sources and the values still to be checked are in [hardware-dados.md](hardware-dados.md).

# ESP32 layer

Runs the portable core of `firmware/core` (`darwin_control`) on a robot exported
by `experiments/export.py`: the control loop on the servo bus, a web page with a
joystick, a serial console and, in the second environment, a micro-ROS node.

**Untested on hardware.** No robot exists yet: this layer compiles and follows
the STS3215 protocol as documented, but nobody has run it on a servo. Calibrate
on the bench with the wheels off the ground before letting it walk.

## Wiring

| Signal | ESP32 pin | Notes |
|---|---|---|
| UART2 TX | GPIO 17 | to the half-duplex adapter's TX input |
| UART2 RX | GPIO 16 | from the adapter's bus side |
| Direction | GPIO 4 | adapter output enable: HIGH transmits, LOW listens |

The servos share one half-duplex TTL bus (all servos on one data wire, each with
its own ID, 1 Mbps, 8N1). Use a 74LVC1G125/74HCT126 with its output enable on
GPIO 4 (the classic Dynamixel circuit); the ESP32 drives the pin around every
packet, so the bus is released before the servo answers. Use the 74HCT part if
the bus idles at 5 V. A "one resistor and a diode" adapter may work at low servo
counts but corrupts answers at 1 Mbps.

Power the servos from the 3S LiPo directly (the 12 V STS3215 takes 6 to 12.6 V)
and the ESP32 from a step-down on the same pack. Give every servo its own ID
before it goes on the bus: all of them ship as ID 1. The firmware sets each
servo's mode from `control.json` (position for joints, wheel for wheels) and
enables torque at boot.

## Build and upload

```bash
PLATFORMIO_CORE_DIR=$PWD/.pio-core pio run -d firmware/esp32 -e esp32 -t upload
```

Put the `control.json` of the export package in `firmware/esp32/data/` next to
`index.html`, then upload the filesystem (the page and the control file):

```bash
PLATFORMIO_CORE_DIR=$PWD/.pio-core pio run -d firmware/esp32 -e esp32 -t uploadfs
```

Without one `pio run -d firmware/esp32 -e esp32 -t upload` builds to
`.pio/build/esp32/firmware.bin` only.

`/control.json` is format 5 of `docs/formato-controle.md` (formats 2 to 4 are
read too, with the missing genes at zero). If it is missing or invalid the
firmware prints one line with the reason at 115200 baud and the robot stays
still: no torque is enabled and no control task starts.

## Wi-Fi and the page

Without `/wifi.json` the robot is its own access point, `darwinbot-XXXX` (the
last two bytes of the MAC), at `http://192.168.4.1/`. To join an existing
network instead, add `firmware/esp32/data/wifi.json` (uploaded with the same
`uploadfs`):

```json
{"ssid": "my-network", "password": "secret",
 "ros_agent_ip": "192.168.1.20", "ros_agent_port": 8888}
```

The page (`data/index.html`, everything inline, no external download) has a
touch joystick (left is negative, as `docs/formato-controle.md` defines), a
speed slider and a stop button, and shows the load and phase of every servo and
the battery voltage, pushed at 10 Hz over a WebSocket on port 81:

```bash
curl http://192.168.4.1/            # the page
```

Nothing on the page blocks the control loop: the control task runs at 50 Hz on
core 1, the web and serial on core 0.

## Serial console (115200)

| Command | Effect |
|---|---|
| `cmd <c>` | steering, -1 to 1 (negative turns left) |
| `speed <s>` | speed, 0 to 1 (1 is the gait as evolved) |
| `stop` | steering straight, slowest gait |
| `status` | robot, rates, command, battery, load/target/phase of every servo |

If no transport sends anything for one second, the firmware ramps the steering
and the speed back to the stop state, so a dropped Wi-Fi link does not leave the
robot walking away. Send `status` from the serial console to check.

## ROS 2 (`-e esp32-ros`)

The same firmware plus a micro-ROS node `darwinbot`, its own task on core 0:

- subscribes `geometry_msgs/Twist` on `cmd_vel`: `angular.z / MAX_YAW_RATE`
  becomes the steering command with the sign flipped (positive z turns left, so
  the command is negative), `linear.x / MAX_SPEED` becomes the speed, clamped to
  0 to 1. Both scales are constants at the top of `src/ros.cpp`; measure the
  robot on the bench and set them.
- publishes `sensor_msgs/JointState` on `darwinbot/joint_states` at 10 Hz:
  `position` is the target the core commands (rad, or rad/s on a wheel) and
  `effort` the signed load over the maximum torque.

It only starts when the robot joined a network (`/wifi.json` with an agent
address). On the computer:

```bash
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888
```

ROS 1 users run `ros1_bridge` on the computer instead of a micro-ROS ROS 2
client. Without an agent the task retries every second and nothing else waits.

```bash
PLATFORMIO_CORE_DIR=$PWD/.pio-core pio run -d firmware/esp32 -e esp32-ros
```

`micro_ros_platformio` downloads a prebuilt micro-ROS library at the first build
and needs network access; the first build of this environment takes minutes.

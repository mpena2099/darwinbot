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

Many adapters loop every transmitted byte back to RX (the echo). The firmware
expects that by default (`-DDARWIN_BUS_ECHO=1`): after each packet it discards
exactly the bytes it sent before reading the answer. If your adapter keeps RX
silent while transmitting, build with `-DDARWIN_BUS_ECHO=0`, or the first bytes
of every answer are lost. A status packet whose length does not match the
request is rejected.

Power the servos from the 3S LiPo directly (the 12 V STS3215 takes 6 to 12.6 V)
and the ESP32 from a step-down on the same pack. Give every servo its own ID
before it goes on the bus: all of them ship as ID 1. At boot the firmware sets each
servo's mode from `control.json` (position for joints, wheel for wheels), writes
the EEPROM registers only when the value differs, limits the torque to
`load_full_scale` of the maximum, reads every joint's present position and takes
it as the goal, and only then enables torque, so nothing jumps.

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
last two bytes of the MAC), at `http://192.168.4.1/`, protected with WPA2. The
default password is `darwinbot` and is printed on the serial console at boot.
**Change it** before anyone else is in range: add `"ap_password": "..."` (8 to 63
characters) to `/wifi.json`. To join an existing
network instead, add `firmware/esp32/data/wifi.json` (uploaded with the same
`uploadfs`):

```json
{"ssid": "my-network", "password": "secret",
 "ros_agent_ip": "192.168.1.20", "ros_agent_port": 8888,
 "ap_password": "something-else"}
```

`ap_password` also works in a file with no `ssid` (access point only).

The page (`data/index.html`, everything inline, no external download) has a
touch joystick (left is negative, as `docs/formato-controle.md` defines), a
speed slider (it starts at 0, so the robot stays stopped when you connect) and a
stop button, and repeats its command every 200 ms while the link is open. It shows the load and phase of every servo and
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
| `stop` | enter STOPPED (see below) |
| `status` | robot, rates, command, battery, load/target/phase of every servo |

Non-finite values (`nan`, `inf`) are rejected on every transport.

## Run state and safety

The robot boots STOPPED and stays so until the first command. STOPPED holds every
joint at its last goal (the present position at boot), keeps the wheels at speed
0 and does not run the controller. Steering with speed 0 turns the robot at its
slowest gait; the robot stops only when steering and speed are both 0. These
enter STOPPED:

- the page's STOP button, serial `stop`, steering and speed both 0 (a ROS
  `cmd_vel` with `linear.x` and `angular.z` both 0);
- no command from any transport for 1 s (a dropped Wi-Fi link);
- a status error byte with the overheat, overcurrent or overload bit from any
  servo: that servo is reported on the serial console and the robot stays
  STOPPED until reboot;
- the battery below 9.6 V (3S): torque is switched off on every servo, and the
  next command after the voltage recovers (10.2 V) re-enables torque from the
  present positions.

Leaving STOPPED resets the controller, so its ramp starts again. While it
ramps (`ramp_time` of the control file) the joints move at 20 % of
`servo_speed`; after that at `servo_speed` itself. A task watchdog on the control
task (2 s) reboots the board if the loop hangs; the servos keep their last goal.
`status` shows whether the robot is running or STOPPED, and why.

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
and needs network access; the first build of this environment takes minutes. It
also builds with the Python of PlatformIO's own environment, which needs these
packages (install them once; CI does the same):

```bash
export PLATFORMIO_CORE_DIR=$PWD/.pio-core
python -m venv $PLATFORMIO_CORE_DIR/penv
$PLATFORMIO_CORE_DIR/penv/bin/pip install platformio catkin_pkg lark-parser \
    empy==3.3.4 colcon-common-extensions importlib-resources pyyaml pytz markupsafe==2.0.1
```

`ros_agent_ip` must be an IPv4 address; without it the node does not start. An
`cmd_vel` with `linear.x` and `angular.z` both 0 is a stop.

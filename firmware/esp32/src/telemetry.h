// A snapshot of the control loop for the page and the ROS publisher: the
// control task writes it, the slower tasks copy it, so neither waits on a socket.
#pragma once

#include "bus.h"

namespace darwinbot_esp32 {

struct Telemetry {
    int n = 0;
    float command = 0.0f;
    float speed = 0.0f;
    float volts = 0.0f;
    bool running = false;     // false: STOPPED
    bool fault = false;       // a servo reported overload or overheat
    bool undervolt = false;   // battery below the 3S limit
    float load[kMaxServos] = {};     // signed, over the maximum torque
    float target[kMaxServos] = {};   // rad (wheel: rad/s), as the core commands
    float phase[kMaxServos] = {};    // rad
};

void telemetry_publish(const Telemetry& t);
void telemetry_copy(Telemetry& t);

}  // namespace darwinbot_esp32

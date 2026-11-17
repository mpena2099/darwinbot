// micro-ROS node "darwinbot" (environment esp32-ros only).
#pragma once

#include "config.h"

namespace darwinbot_esp32 {

// Its own task on core 0 (16 kB stack): it never blocks the control loop.
void ros_begin(const Config& cfg);

}  // namespace darwinbot_esp32

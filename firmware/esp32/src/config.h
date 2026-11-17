// /control.json of docs/formato-controle.md, read once at boot into the
// parameters of the portable core.
#pragma once

#include <cstddef>
#include <cstdint>

#include "bus.h"
#include "darwin_control.h"

namespace darwinbot_esp32 {

struct Config {
    char robot[24] = "";
    int format = 0;
    float frequency = 0.0f;       // Hz
    float ramp_time = 0.0f;       // s
    float control_hz = 50.0f;
    float load_full_scale = darwinbot::Controller::kLoadFullScale;
    float servo_speed = darwinbot::Controller::kServoSpeed;
    int n = 0;                    // servos, in layout order
    uint8_t ids[kMaxServos] = {};
    bool wheel[kMaxServos] = {};
    char names[kMaxServos][16] = {};
    darwinbot::ServoParams params[kMaxServos] = {};
};

// Loads /control.json (format 5; 2 to 4 with the newer genes off). Returns
// false and writes a one-line reason in err, leaving cfg untouched.
bool load_config(Config& cfg, char* err, size_t err_len);

}  // namespace darwinbot_esp32

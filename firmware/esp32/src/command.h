// One command interface for every transport (web page, WebSocket, serial, ROS).
// The control task reads it once per tick; nothing here blocks or allocates.
#pragma once

namespace cmd {

constexpr unsigned long kTimeoutMs = 1000;   // no command for 1 s -> ramp to stop
constexpr float kFallbackSeconds = 0.5f;     // time the fallback ramp takes

// Steering in [-1, 1]: 0 straight, negative turns left, positive right.
void set_command(float c);

// Speed in [0, 1]: 1 is the gait as evolved, 0 the slowest one learned.
void set_speed(float s);

// Steering straight and the slowest gait, at once.
void stop();

// What the control loop should apply now, after the fallback ramp.
void read(float& c, float& s);

// Called once per control tick: ramps to stop when no transport has sent
// anything for kTimeoutMs. Thread-safe (atomics only).
void upkeep(float dt);

}  // namespace cmd

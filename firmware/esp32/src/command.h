// One command interface for every transport (web page, WebSocket, serial, ROS).
// The control task reads it once per tick; nothing here blocks or allocates.
// Command, speed, the time stamp and the run state live under one critical
// section, so a reader never sees half of an update.
#pragma once

namespace cmd {

constexpr unsigned long kTimeoutMs = 1000;   // no command for 1 s -> STOPPED

// Steering in [-1, 1]: 0 straight, negative turns left, positive right.
// Returns false (and changes nothing) for a non-finite value.
bool set_command(float c);

// Speed in [0, 1]: 1 is the gait as evolved, 0 the slowest one learned.
// Returns false (and changes nothing) for a non-finite value.
bool set_speed(float s);

// Both at once, one critical section (ROS Twist).
bool set(float c, float s);

// Enter STOPPED: the robot holds its pose, wheels at rest.
void stop();

struct State {
    float command = 0.0f;
    float speed = 0.0f;
    bool running = false;   // false: STOPPED
};

// The state to apply now. Boots STOPPED. Any command or speed other than zero
// leaves STOPPED; both zero, stop() or no transport for kTimeoutMs enters it.
State read();

}  // namespace cmd

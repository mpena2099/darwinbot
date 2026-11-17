#include "command.h"

#include <Arduino.h>

#include <atomic>

namespace cmd {
namespace {

std::atomic<float> g_command{0.0f};
std::atomic<float> g_speed{0.0f};
std::atomic<unsigned long> g_stamp{0};   // millis() of the last command
bool g_stamped = false;                  // no transport has sent anything yet

float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void stamp() {
    g_stamp.store(millis());
    g_stamped = true;
}

}  // namespace

void set_command(float c) {
    g_command.store(clampf(c, -1.0f, 1.0f));
    stamp();
}

void set_speed(float s) {
    g_speed.store(clampf(s, 0.0f, 1.0f));
    stamp();
}

void stop() {
    g_command.store(0.0f);
    g_speed.store(0.0f);
    stamp();
}

void read(float& c, float& s) {
    c = g_command.load();
    s = g_speed.load();
}

void upkeep(float dt) {
    // With no transport at all the robot waits at the ramped-to-stop state.
    if (g_stamped && millis() - g_stamp.load() < kTimeoutMs) return;
    const float step = dt / kFallbackSeconds;
    const float c = g_command.load();
    const float s = g_speed.load();
    g_command.store(c > 0.0f ? (c - step > 0.0f ? c - step : 0.0f)
                             : (c + step < 0.0f ? c + step : 0.0f));
    g_speed.store(s > step ? s - step : 0.0f);
}

}  // namespace cmd

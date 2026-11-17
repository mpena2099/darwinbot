#include "command.h"

#include <Arduino.h>

#include <cmath>

namespace cmd {
namespace {

portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;
float g_command = 0.0f;
float g_speed = 0.0f;
unsigned long g_stamp = 0;   // millis() of the last command
bool g_running = false;      // boots STOPPED until the first command

float clampf(float v, float lo, float hi) {
    if (!std::isfinite(v)) return 0.0f;
    return v < lo ? lo : (v > hi ? hi : v);
}

// Caller holds g_mux.
void commit() {
    g_stamp = millis();
    g_running = g_speed != 0.0f;
}

}  // namespace

bool set_command(float c) {
    if (!std::isfinite(c)) return false;
    portENTER_CRITICAL(&g_mux);
    g_command = clampf(c, -1.0f, 1.0f);
    commit();
    portEXIT_CRITICAL(&g_mux);
    return true;
}

bool set_speed(float s) {
    if (!std::isfinite(s)) return false;
    portENTER_CRITICAL(&g_mux);
    g_speed = clampf(s, 0.0f, 1.0f);
    commit();
    portEXIT_CRITICAL(&g_mux);
    return true;
}

bool set(float c, float s) {
    if (!std::isfinite(c) || !std::isfinite(s)) return false;
    portENTER_CRITICAL(&g_mux);
    g_command = clampf(c, -1.0f, 1.0f);
    g_speed = clampf(s, 0.0f, 1.0f);
    commit();
    portEXIT_CRITICAL(&g_mux);
    return true;
}

void stop() {
    portENTER_CRITICAL(&g_mux);
    g_command = 0.0f;
    g_speed = 0.0f;
    g_stamp = millis();
    g_running = false;
    portEXIT_CRITICAL(&g_mux);
}

State read() {
    State st;
    portENTER_CRITICAL(&g_mux);
    if (g_running && millis() - g_stamp >= kTimeoutMs) {
        g_command = 0.0f;
        g_speed = 0.0f;
        g_running = false;
    }
    st.command = g_command;
    st.speed = g_speed;
    st.running = g_running;
    portEXIT_CRITICAL(&g_mux);
    return st;
}

}  // namespace cmd

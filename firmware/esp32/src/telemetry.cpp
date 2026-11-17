#include "telemetry.h"

#include <Arduino.h>
#include <cstring>

namespace darwinbot_esp32 {
namespace {

portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;
Telemetry g_snapshot;

}  // namespace

void telemetry_publish(const Telemetry& t) {
    portENTER_CRITICAL(&g_mux);
    std::memcpy(&g_snapshot, &t, sizeof(t));
    portEXIT_CRITICAL(&g_mux);
}

void telemetry_copy(Telemetry& t) {
    portENTER_CRITICAL(&g_mux);
    std::memcpy(&t, &g_snapshot, sizeof(t));
    portEXIT_CRITICAL(&g_mux);
}

}  // namespace darwinbot_esp32

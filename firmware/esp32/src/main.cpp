// ESP32 layer: firmware/core (darwin_control) on a robot from experiments/export.py.
// Untested on hardware: no robot exists yet.
#include <Arduino.h>
#include <LittleFS.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bus.h"
#include "command.h"
#include "config.h"
#include "darwin_control.h"
#include "telemetry.h"
#include "web.h"
#ifdef DARWIN_ROS
#include "ros.h"
#endif

namespace darwinbot_esp32 {
namespace {

Config g_cfg;
Bus g_bus;
darwinbot::Controller* g_controller = nullptr;
char g_error[192] = "no control file";

// The control loop: 50 Hz on core 1, its own task, never waiting on a socket.
void control_task(void*) {
    const int n = g_cfg.n;
    float loads[kMaxServos] = {};
    uint8_t failures[kMaxServos] = {};
    bool offline[kMaxServos] = {};

    uint8_t position_ids[kMaxServos], wheel_ids[kMaxServos];
    int position_at[kMaxServos], wheel_at[kMaxServos];
    uint16_t position_steps[kMaxServos], wheel_steps[kMaxServos];
    int n_position = 0, n_wheel = 0;
    for (int i = 0; i < n; ++i) {
        if (g_cfg.wheel[i]) {
            wheel_ids[n_wheel] = g_cfg.ids[i];
            wheel_at[n_wheel++] = i;
        } else {
            position_ids[n_position] = g_cfg.ids[i];
            position_at[n_position++] = i;
        }
    }

    const float dt = 1.0f / g_cfg.control_hz;
    const TickType_t period = std::max<TickType_t>(1, pdMS_TO_TICKS(1000.0f / g_cfg.control_hz));
    TickType_t wake = xTaskGetTickCount();
    Telemetry t;
    t.n = n;
    uint32_t tick = 0;

    for (;;) {
        cmd::upkeep(dt);
        float c = 0.0f, s = 0.0f;
        cmd::read(c, s);
        g_controller->set_command(c);
        g_controller->set_speed(s);

        // Present load, one servo at a time. A servo that stops answering is
        // polled once a second instead of holding the tick.
        for (int i = 0; i < n; ++i) {
            loads[i] = 0.0f;
            if (offline[i]) {
                if (tick & 63) continue;
                offline[i] = false;
            }
            float load = 0.0f;
            if (g_bus.read_load(g_cfg.ids[i], load)) {
                loads[i] = load;
                failures[i] = 0;
            } else if (++failures[i] >= 3) {
                offline[i] = true;
            }
        }
        if ((tick & 31) == 0) {   // battery from the first servo, ~1.5 Hz at 50 Hz
            float volts = 0.0f;
            if (g_bus.read_voltage(g_cfg.ids[0], volts)) t.volts = volts;
        }

        g_controller->tick(loads);

        for (int k = 0; k < n_position; ++k) {
            position_steps[k] =
                static_cast<uint16_t>(std::clamp(g_controller->steps[position_at[k]], 0, 4095));
        }
        g_bus.write_regs(kRegGoalPosition, position_ids, position_steps, n_position);

        for (int k = 0; k < n_wheel; ++k) {
            const float speed = g_controller->wheel_speeds[wheel_at[k]] * kStepsPerRad;
            int32_t raw = static_cast<int32_t>(std::lround(std::fabs(speed)));
            if (raw > 0x7FFF) raw = 0x7FFF;
            if (speed < 0.0f) raw |= 0x8000;   // bit 15 is the direction
            wheel_steps[k] = static_cast<uint16_t>(raw);
        }
        g_bus.write_regs(kRegGoalSpeed, wheel_ids, wheel_steps, n_wheel);

        t.command = c;
        t.speed = s;
        for (int i = 0; i < n; ++i) {
            t.load[i] = loads[i];
            t.target[i] = g_controller->targets[i];
            t.phase[i] = g_controller->phases()[i];
        }
        telemetry_publish(t);

        ++tick;
        vTaskDelayUntil(&wake, period);
    }
}

void print_status() {
    if (!g_controller) {
        Serial.printf("no controller: %s\n", g_error);
        return;
    }
    Telemetry t;
    telemetry_copy(t);
    Serial.printf("%s, %d servos, %.0f Hz, format %d, command %.2f, speed %.2f, battery %.1f V\n",
                  g_cfg.robot, g_cfg.n, g_cfg.control_hz, g_cfg.format, t.command, t.speed, t.volts);
    for (int i = 0; i < g_cfg.n; ++i) {
        Serial.printf("  %3d %-12s %-8s load %+.2f target %+.2f phase %.2f\n", g_cfg.ids[i],
                      g_cfg.names[i], g_cfg.wheel[i] ? "wheel" : "position", t.load[i],
                      t.target[i], t.phase[i]);
    }
}

void run_command(char* line) {
    float value = 0.0f;
    if (std::strncmp(line, "cmd ", 4) == 0 && sscanf(line + 4, "%f", &value) == 1) {
        cmd::set_command(value);
        Serial.printf("cmd %.2f\n", value);
    } else if (std::strncmp(line, "speed ", 6) == 0 && sscanf(line + 6, "%f", &value) == 1) {
        cmd::set_speed(value);
        Serial.printf("speed %.2f\n", value);
    } else if (std::strcmp(line, "stop") == 0) {
        cmd::stop();
        Serial.println("stop");
    } else if (std::strcmp(line, "status") == 0) {
        print_status();
    } else if (std::strcmp(line, "help") == 0) {
        Serial.println("cmd <-1..1> | speed <0..1> | stop | status");
    } else {
        Serial.printf("unknown '%s', try: cmd <-1..1> | speed <0..1> | stop | status\n", line);
    }
}

void serial_loop() {
    static char line[48];
    static size_t len = 0;
    while (Serial.available()) {
        const char ch = static_cast<char>(Serial.read());
        if (ch == '\n' || ch == '\r') {
            if (len) {
                line[len] = '\0';
                run_command(line);
                len = 0;
            }
        } else if (len < sizeof(line) - 1) {
            line[len++] = ch;
        }
    }
}

void start_controller() {
    if (!load_config(g_cfg, g_error, sizeof(g_error))) {
        Serial.printf("ERROR %s\nThe robot stays still.\n", g_error);
        return;
    }
    Serial.printf("control: %s, %d servos, %.0f Hz, format %d\n", g_cfg.robot, g_cfg.n,
                  g_cfg.control_hz, g_cfg.format);

    g_bus.begin();
    for (int i = 0; i < g_cfg.n; ++i) g_bus.set_mode(g_cfg.ids[i], g_cfg.wheel[i]);

    std::vector<darwinbot::ServoParams> params(g_cfg.params, g_cfg.params + g_cfg.n);
    g_controller = new darwinbot::Controller(g_cfg.frequency, g_cfg.ramp_time, g_cfg.control_hz,
                                             params, g_cfg.load_full_scale, g_cfg.servo_speed);
    g_error[0] = '\0';
    xTaskCreatePinnedToCore(control_task, "control", 4096, nullptr, 5, nullptr, 1);
}

}  // namespace
}  // namespace darwinbot_esp32

void setup() {
    using namespace darwinbot_esp32;
    Serial.begin(115200);
    delay(300);
    Serial.println("\nDarwinBot ESP32, control core from firmware/core (untested on hardware)");

    if (!LittleFS.begin(true)) {
        Serial.println("ERROR LittleFS did not mount");
    }
    start_controller();

    net_begin();
    web_begin();
#ifdef DARWIN_ROS
    if (g_controller && net_config().station) ros_begin(g_cfg);
#endif
    Serial.println("commands: cmd <-1..1> | speed <0..1> | stop | status");
}

void loop() {
    darwinbot_esp32::serial_loop();
    darwinbot_esp32::web_loop();
    delay(1);
}

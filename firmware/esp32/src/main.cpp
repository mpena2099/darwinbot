// ESP32 layer: firmware/core (darwin_control) on a robot from experiments/export.py.
// Untested on hardware: no robot exists yet.
#include <Arduino.h>
#include <LittleFS.h>
#include <esp_task_wdt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
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
// The controller lives in static storage, built once at boot. Leaving STOPPED
// resets it, so its ramp starts again. Nothing is allocated per tick.
alignas(darwinbot::Controller) unsigned char g_controller_storage[sizeof(darwinbot::Controller)];
darwinbot::Controller* g_controller = nullptr;
char g_error[192] = "no control file";

// Servo state shared by boot and the control task (the task starts after boot).
uint16_t g_hold[kMaxServos] = {};     // joint goal held while STOPPED, in steps
bool g_torque_on[kMaxServos] = {};    // false: the servo did not answer or was shut off
uint16_t g_speed_full = 0;            // position move speed limit, steps/s
uint16_t g_speed_slow = 0;            // 20 % of it, while the controller ramps in

constexpr float kUndervoltage = 9.6f;       // V, 3S pack at 3.2 V per cell
constexpr float kUndervoltageClear = 10.2f; // V, hysteresis before torque may return
constexpr float kRampSpeedFraction = 0.2f;

void build_controller() {
    if (g_controller) g_controller->~Controller();
    std::vector<darwinbot::ServoParams> params(g_cfg.params, g_cfg.params + g_cfg.n);
    g_controller = new (g_controller_storage)
        darwinbot::Controller(g_cfg.frequency, g_cfg.ramp_time, g_cfg.control_hz, params,
                              g_cfg.load_full_scale, g_cfg.servo_speed, g_cfg.clip_mode);
}

// Torque on for every servo that has none, each one starting from where it is.
void arm_servos() {
    for (int i = 0; i < g_cfg.n; ++i) {
        if (g_torque_on[i]) continue;
        uint16_t position = 0;
        if (g_bus.arm(g_cfg.ids[i], g_cfg.wheel[i], g_speed_slow, position)) {
            g_torque_on[i] = true;
            if (!g_cfg.wheel[i]) g_hold[i] = position;
        } else {
            Serial.printf("servo %d (%s) did not answer: torque stays off\n", g_cfg.ids[i],
                          g_cfg.names[i]);
        }
    }
}

void torque_off_all() {
    for (int i = 0; i < g_cfg.n; ++i) {
        g_bus.torque(g_cfg.ids[i], false);
        g_torque_on[i] = false;
    }
}

// The control loop: 50 Hz on core 1, its own task, never waiting on a socket.
void control_task(void*) {
    const int n = g_cfg.n;
    float loads[kMaxServos] = {};
    uint8_t failures[kMaxServos] = {};
    bool offline[kMaxServos] = {};
    bool faulted[kMaxServos] = {};   // overload or overheat: stays until reboot
    bool any_fault = false;
    bool undervolt = false;

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
    uint16_t speed_values[kMaxServos];

    const TickType_t period = std::max<TickType_t>(1, pdMS_TO_TICKS(1000.0f / g_cfg.control_hz));
    const int ramp_ticks = static_cast<int>(std::ceil(g_cfg.ramp_time * g_cfg.control_hz)) + 1;
    TickType_t wake = xTaskGetTickCount();
    Telemetry t;
    t.n = n;
    uint32_t tick = 0;
    bool was_running = false;
    int ramp_left = 0;

    esp_task_wdt_init(2, true);   // a stuck loop reboots the board, servos hold their goal
    esp_task_wdt_add(nullptr);

    for (;;) {
        esp_task_wdt_reset();
        cmd::State st = cmd::read();

        // Present load, one servo at a time. A servo that stops answering is
        // polled once every 64 ticks (staggered per servo) instead of holding
        // the tick.
        for (int i = 0; i < n; ++i) {
            loads[i] = 0.0f;
            if (offline[i]) {
                if (((tick + i) & 63) != 0) continue;
                offline[i] = false;
            }
            float load = 0.0f;
            if (g_bus.read_load(g_cfg.ids[i], load)) {
                loads[i] = load;
                failures[i] = 0;
                if ((g_bus.last_error() & kErrorMask) && !faulted[i]) {
                    faulted[i] = true;
                    any_fault = true;
                    Serial.printf("servo %d (%s) reports error 0x%02X: STOPPED until reboot\n",
                                  g_cfg.ids[i], g_cfg.names[i], g_bus.last_error());
                }
            } else if (++failures[i] >= 3) {
                offline[i] = true;
            }
        }
        if ((tick & 31) == 0) {   // battery from the first online servo, ~1.5 Hz at 50 Hz
            for (int i = 0; i < n; ++i) {
                float volts = 0.0f;
                if (offline[i] || !g_bus.read_voltage(g_cfg.ids[i], volts)) continue;
                t.volts = volts;
                if (volts < kUndervoltage && !undervolt) {
                    undervolt = true;
                    torque_off_all();
                    Serial.printf("battery %.1f V below %.1f V: torque off, STOPPED\n", volts,
                                  kUndervoltage);
                } else if (volts >= kUndervoltageClear) {
                    undervolt = false;
                }
                break;
            }
        }
        if (any_fault || undervolt) cmd::stop();

        const bool running = st.running && !any_fault && !undervolt;
        if (running && !was_running) {
            arm_servos();            // servos shut off by an undervoltage start from where they are
            g_controller->reset();   // the ramp starts again
            ramp_left = ramp_ticks;
            for (int k = 0; k < n_position; ++k) speed_values[k] = g_speed_slow;
            g_bus.write_regs(kRegGoalSpeed, position_ids, speed_values, n_position);
        }
        was_running = running;

        if (running) {
            g_controller->set_command(st.command);
            g_controller->set_speed(st.speed);
            g_controller->tick(loads);
            for (int k = 0; k < n_position; ++k) {
                g_hold[position_at[k]] =
                    static_cast<uint16_t>(std::clamp(g_controller->steps[position_at[k]], 0, 4095));
            }
            if (ramp_left > 0 && --ramp_left == 0) {   // ramp over: the evolved speed limit
                for (int k = 0; k < n_position; ++k) speed_values[k] = g_speed_full;
                g_bus.write_regs(kRegGoalSpeed, position_ids, speed_values, n_position);
            }
        }

        // STOPPED holds the last joint goal and keeps the wheels at rest.
        for (int k = 0; k < n_position; ++k) position_steps[k] = g_hold[position_at[k]];
        g_bus.write_regs(kRegGoalPosition, position_ids, position_steps, n_position);

        for (int k = 0; k < n_wheel; ++k) {
            const float speed = running ? g_controller->wheel_speeds[wheel_at[k]] * kStepsPerRad : 0.0f;
            int32_t raw = static_cast<int32_t>(std::lround(std::fabs(speed)));
            if (raw > 0x7FFF) raw = 0x7FFF;
            if (speed < 0.0f) raw |= 0x8000;   // bit 15 is the direction
            wheel_steps[k] = static_cast<uint16_t>(raw);
        }
        g_bus.write_regs(kRegGoalSpeed, wheel_ids, wheel_steps, n_wheel);

        t.command = st.command;
        t.speed = st.speed;
        t.running = running;
        t.fault = any_fault;
        t.undervolt = undervolt;
        for (int i = 0; i < n; ++i) {
            t.load[i] = loads[i];
            t.target[i] = running ? g_controller->targets[i] : 0.0f;
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
    Serial.printf("%s, %d servos, %.0f Hz, format %d, %s, command %.2f, speed %.2f, battery %.1f V\n",
                  g_cfg.robot, g_cfg.n, g_cfg.control_hz, g_cfg.format,
                  t.fault ? "STOPPED (servo error)" : t.undervolt ? "STOPPED (battery low)"
                  : t.running ? "running" : "STOPPED",
                  t.command, t.speed, t.volts);
    for (int i = 0; i < g_cfg.n; ++i) {
        Serial.printf("  %3d %-12s %-8s load %+.2f target %+.2f phase %.2f\n", g_cfg.ids[i],
                      g_cfg.names[i], g_cfg.wheel[i] ? "wheel" : "position", t.load[i],
                      t.target[i], t.phase[i]);
    }
}

void run_command(char* line) {
    float value = 0.0f;
    if (std::strncmp(line, "cmd ", 4) == 0 && sscanf(line + 4, "%f", &value) == 1) {
        if (cmd::set_command(value)) Serial.printf("cmd %.2f\n", value);
        else Serial.println("rejected: not a finite number");
    } else if (std::strncmp(line, "speed ", 6) == 0 && sscanf(line + 6, "%f", &value) == 1) {
        if (cmd::set_speed(value)) Serial.printf("speed %.2f\n", value);
        else Serial.println("rejected: not a finite number");
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

    g_speed_full = static_cast<uint16_t>(
        std::min(32767L, std::lround(g_cfg.servo_speed * kStepsPerRad)));
    g_speed_slow = std::max<uint16_t>(
        1, static_cast<uint16_t>(g_speed_full * kRampSpeedFraction));

    g_bus.begin();
    for (int i = 0; i < g_cfg.n; ++i) {
        const uint8_t id = g_cfg.ids[i];
        if (!g_bus.set_mode(id, g_cfg.wheel[i])) {
            Serial.printf("servo %d (%s) did not answer\n", id, g_cfg.names[i]);
            continue;
        }
        g_bus.set_torque_limit(id, g_cfg.load_full_scale);
    }
    arm_servos();   // present position becomes the goal before the torque comes on
    build_controller();
    g_error[0] = '\0';
    xTaskCreatePinnedToCore(control_task, "control", 6144, nullptr, 5, nullptr, 1);
}

}  // namespace
}  // namespace darwinbot_esp32

void setup() {
    using namespace darwinbot_esp32;
    Serial.begin(115200);
    delay(300);
    Serial.println("\nDarwinBot ESP32, control core from firmware/core (untested on hardware)");

    // Never format: a failed mount must not erase control.json and wifi.json.
    if (!LittleFS.begin(false)) {
        Serial.println("ERROR LittleFS did not mount (not formatted: run 'pio run -t uploadfs')");
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

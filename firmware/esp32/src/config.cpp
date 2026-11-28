#include "config.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <LittleFS.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace darwinbot_esp32 {
namespace {

bool fail(char* err, size_t err_len, const char* why) {
    snprintf(err, err_len, "%s", why);
    return false;
}

}  // namespace

bool load_config(Config& cfg, char* err, size_t err_len) {
    File f = LittleFS.open("/control.json", "r");
    if (!f) {
        return fail(err, err_len,
                    "no /control.json on LittleFS: put the export package's file "
                    "in firmware/esp32/data/ and run 'pio run -t uploadfs'");
    }
    JsonDocument doc;
    const DeserializationError broken = deserializeJson(doc, f);
    f.close();
    if (broken) {
        snprintf(err, err_len, "/control.json is not JSON: %s", broken.c_str());
        return false;
    }

    const int format = doc["format"] | 0;
    if (format < 2 || format > 5) {
        snprintf(err, err_len, "/control.json format %d, only 2 to 5 are read", format);
        return false;
    }
    JsonArrayConst servos = doc["servos"].as<JsonArrayConst>();
    const int n = servos.isNull() ? 0 : static_cast<int>(servos.size());
    if (n < 1 || n > kMaxServos) {
        snprintf(err, err_len, "/control.json has %d servos, expected 1 to %d", n, kMaxServos);
        return false;
    }
    if (!doc["frequency"].is<float>() || !doc["ramp_time"].is<float>() ||
        !doc["control_hz"].is<float>() || doc["control_hz"].as<float>() <= 0.0f ||
        !std::isfinite(doc["control_hz"].as<float>())) {
        return fail(err, err_len,
                    "/control.json needs frequency, ramp_time and a positive control_hz");
    }
    if (!std::isfinite(doc["frequency"].as<float>()) || doc["frequency"].as<float>() <= 0.0f ||
        !std::isfinite(doc["ramp_time"].as<float>()) || doc["ramp_time"].as<float>() < 0.0f) {
        return fail(err, err_len,
                    "/control.json needs a finite, positive frequency and a ramp_time >= 0");
    }

    Config c;
    c.format = format;
    c.n = n;
    c.frequency = doc["frequency"].as<float>();
    c.ramp_time = doc["ramp_time"].as<float>();
    c.control_hz = doc["control_hz"].as<float>();
    c.load_full_scale = doc["load_full_scale"] | darwinbot::Controller::kLoadFullScale;
    c.servo_speed = doc["servo_speed"] | darwinbot::Controller::kServoSpeed;
    if (!std::isfinite(c.load_full_scale) || !std::isfinite(c.servo_speed) ||
        c.load_full_scale <= 0.0f || c.load_full_scale > 1.0f || c.servo_speed <= 0.0f) {
        return fail(err, err_len,
                    "/control.json needs load_full_scale in (0, 1] and a positive servo_speed");
    }
    const char* clip = doc["clip_mode"] | "command";
    if (std::strcmp(clip, "headroom") == 0) {
        c.clip_mode = darwinbot::Controller::ClipMode::Headroom;
    } else if (std::strcmp(clip, "command") != 0) {
        snprintf(err, err_len, "/control.json clip_mode '%s', expected command or headroom",
                 clip);
        return false;
    }
    strlcpy(c.robot, doc["robot"] | "robot", sizeof(c.robot));

    // Required fields, so a truncated file never turns into a robot that moves
    // with silent zeros.
    auto need = [&](JsonObjectConst s, int id, const char* key) {
        if (s[key].is<float>() || s[key].is<int>()) return true;
        snprintf(err, err_len, "/control.json: servo %d has no '%s'", id, key);
        return false;
    };

    for (int i = 0; i < n; ++i) {
        JsonObjectConst s = servos[i].as<JsonObjectConst>();
        const int id = s["id"] | 0;
        if (id < 1 || id > 253) {
            snprintf(err, err_len, "/control.json: servo %d, id outside 1..253", id);
            return false;
        }
        for (int j = 0; j < i; ++j) {
            if (c.ids[j] == id) {
                snprintf(err, err_len, "/control.json: servo id %d twice", id);
                return false;
            }
        }
        c.ids[i] = static_cast<uint8_t>(id);
        strlcpy(c.names[i], s["name"] | "servo", sizeof(c.names[i]));

        darwinbot::ServoParams& p = c.params[i];
        const char* mode = s["mode"] | "";
        if (std::strcmp(mode, "wheel") == 0) {
            c.wheel[i] = true;
            p.mode = darwinbot::ServoParams::Mode::Wheel;
            if (!need(s, id, "speed") || !need(s, id, "sign")) return false;
            p.speed = s["speed"].as<float>();
            p.sign = s["sign"].as<float>();
        } else if (std::strcmp(mode, "position") == 0) {
            for (const char* key : {"amp1", "phase1", "offset", "amp2", "phase2", "sigma", "min", "max"}) {
                if (!need(s, id, key)) return false;
            }
            p.amp1 = s["amp1"].as<float>();
            p.phase1 = s["phase1"].as<float>();
            p.offset = s["offset"].as<float>();
            p.amp2 = s["amp2"].as<float>();
            p.phase2 = s["phase2"].as<float>();
            p.sigma = s["sigma"].as<float>();
            p.min = s["min"].as<float>();
            p.max = s["max"].as<float>();
            p.zero = s["zero"] | 2048.0f;   // calibration, counts at 0 rad
            p.sign = s["sign"] | 1.0f;
            if (!(p.min <= p.max)) {
                snprintf(err, err_len, "/control.json: servo %d has min %.3f above max %.3f", id,
                         p.min, p.max);
                return false;
            }
        } else {
            snprintf(err, err_len, "/control.json: servo %d mode '%s', expected position or wheel", id, mode);
            return false;
        }

        // Formats 2 to 4 carry no steering, speed or coupling genes.
        if (format >= 3) {
            p.steer_gain = s["steer_gain"] | 0.0f;
            p.steer_shift = s["steer_shift"] | 0.0f;
        }
        if (format >= 4) {
            p.speed_gain = s["speed_gain"] | 0.0f;
            p.speed_shift = s["speed_shift"] | 0.0f;
        }
        if (format >= 5) {
            p.couple_to = s["couple_to"] | -1;
            if (p.couple_to < -1 || p.couple_to >= n) {
                snprintf(err, err_len, "/control.json: servo %d couple_to %d, expected -1 to %d",
                         id, p.couple_to, n - 1);
                return false;
            }
            p.couple_k = s["couple_k"] | 0.0f;
            p.couple_lag = s["couple_lag"] | 0.0f;
        }
    }

    cfg = c;
    return true;
}

}  // namespace darwinbot_esp32

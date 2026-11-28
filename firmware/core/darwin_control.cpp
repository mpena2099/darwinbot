#include "darwin_control.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace darwinbot {

namespace {
constexpr float kTwoPi = 6.283185307179586f;
constexpr float kStepsPerRad = 4096.0f / kTwoPi;
}  // namespace

Controller::Controller(float frequency, float ramp_time, float control_hz,
                       std::vector<ServoParams> servos, float load_full_scale,
                       float servo_speed, ClipMode clip_mode)
    : omega_(kTwoPi * frequency),
      ramp_time_(ramp_time),
      dt_(control_hz > 0.0f ? 1.0f / control_hz : 0.0f),
      load_full_scale_(load_full_scale > 0.0f ? load_full_scale : kLoadFullScale),
      servo_speed_(servo_speed > 0.0f ? servo_speed : kServoSpeed),
      clip_mode_(clip_mode),
      servos_(std::move(servos)),
      base_(servos_) {
    const size_t n = servos_.size();
    theta_.assign(n, 0.0f);
    targets.assign(n, 0.0f);
    steps.assign(n, 0);
    wheel_speeds.assign(n, 0.0f);
    apply();
}

void Controller::set_command(float c) {
    command_ = std::clamp(c, -1.0f, 1.0f);
    apply();
}

void Controller::set_speed(float s) {
    speed_ = std::clamp(s, 0.0f, 1.0f);
    apply();
}

void Controller::reset() {
    std::fill(theta_.begin(), theta_.end(), 0.0f);
    std::fill(targets.begin(), targets.end(), 0.0f);
    std::fill(steps.begin(), steps.end(), 0);
    std::fill(wheel_speeds.begin(), wheel_speeds.end(), 0.0f);
    time_ = 0.0f;
}

void Controller::apply() {
    const float d = 1.0f - speed_;
    for (size_t i = 0; i < servos_.size(); ++i) {
        const ServoParams& b = base_[i];
        ServoParams& p = servos_[i];
        if (p.mode == ServoParams::Mode::Wheel) {
            p.speed = std::clamp(b.speed + command_ * b.steer_shift +
                                     d * b.speed_shift,
                                 -servo_speed_, servo_speed_);
            continue;
        }
        const float gain =
            std::clamp(1.0f + command_ * b.steer_gain, 0.0f, 2.0f) *
            std::clamp(1.0f - d * b.speed_gain, 0.0f, 2.0f);
        p.offset = b.offset + command_ * b.steer_shift + d * b.speed_shift;
        float a1 = b.amp1;
        float a2 = b.amp2;
        if (clip_mode_ == ClipMode::Headroom) {
            // Scale the base amplitudes by the headroom of the stronger steering
            // side first, so both sides keep their proportion after the commands.
            const float peak_base =
                (b.amp1 + 2.0f * b.amp2) * (omega_ + std::fabs(b.sigma));
            const float g = std::clamp(1.0f + std::fabs(b.steer_gain), 0.0f, 2.0f);
            const float k0 =
                std::min(1.0f, servo_speed_ / std::max(peak_base * g, 1e-12f));
            a1 *= k0;
            a2 *= k0;
        }
        p.amp1 = a1 * gain;
        p.amp2 = a2 * gain;
        // A servo cannot follow a signal whose peak speed exceeds its own.
        // The phase advances at up to omega + |sigma| under load feedback.
        const float peak = (p.amp1 + 2.0f * p.amp2) * (omega_ + std::fabs(b.sigma));
        const float k = std::min(1.0f, servo_speed_ / std::max(peak, 1e-12f));
        p.amp1 *= k;
        p.amp2 *= k;
    }
}

void Controller::tick(const float* loads_normalised) {
    const float ramp =
        ramp_time_ > 0.0f ? std::min(1.0f, time_ / ramp_time_) : 1.0f;
    const float h = dt_ / static_cast<float>(kSubsteps);
    // A body never has more servos than morphology.MAX_SERVOS, so the rates of
    // one substep live in a fixed array: no heap use on the robot.
    const std::size_t n =
        std::min(servos_.size(), static_cast<std::size_t>(kMaxServos));
    std::array<float, kMaxServos> rate{};

    // Targets from the phases at t, before any substep advances them.
    for (std::size_t i = 0; i < n; ++i) {
        const ServoParams& p = servos_[i];

        if (p.mode == ServoParams::Mode::Wheel) {
            targets[i] = ramp * p.speed;
            steps[i] = 0;
            wheel_speeds[i] = ramp * p.speed * p.sign;
            continue;
        }

        const float theta = theta_[i];
        const float target =
            ramp * (p.offset + p.amp1 * std::sin(theta + p.phase1) +
                    p.amp2 * std::sin(2.0f * theta + p.phase2));
        targets[i] = std::clamp(target, p.min, p.max);
        steps[i] = static_cast<int>(
            std::lround(p.zero + p.sign * targets[i] * kStepsPerRad));
        wheel_speeds[i] = 0.0f;
    }

    for (int k = 0; k < kSubsteps; ++k) {
        // Every rate reads the phases from the start of the substep.
        for (std::size_t i = 0; i < n; ++i) {
            const ServoParams& p = servos_[i];
            const float load =
                loads_normalised
                    ? std::clamp(p.sign * loads_normalised[i] /
                                     load_full_scale_,
                                 -1.0f, 1.0f)
                    : 0.0f;
            rate[i] = omega_ - p.sigma * load * std::cos(theta_[i]);
        }
        for (std::size_t j = 0; j < n; ++j) {
            const ServoParams& p = servos_[j];
            if (p.couple_k == 0.0f || p.couple_to < 0 ||
                p.couple_to >= static_cast<int>(n)) {
                continue;
            }
            const std::size_t partner = static_cast<std::size_t>(p.couple_to);
            rate[j] += p.couple_k *
                       std::sin(theta_[partner] - theta_[j] - p.couple_lag);
            rate[partner] +=
                p.couple_k *
                std::sin(theta_[j] - theta_[partner] + p.couple_lag);
        }
        for (std::size_t i = 0; i < n; ++i) theta_[i] += h * rate[i];
    }

    for (std::size_t i = 0; i < n; ++i) {
        // keep the phase small so float precision never stalls it
        theta_[i] = std::fmod(theta_[i], kTwoPi);
        if (theta_[i] < 0.0f) theta_[i] += kTwoPi;
    }

    time_ += dt_;
}

}  // namespace darwinbot

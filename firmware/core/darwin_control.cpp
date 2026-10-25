#include "darwin_control.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace darwinbot {

namespace {
constexpr float kTwoPi = 6.283185307179586f;
constexpr float kStepsPerRad = 4096.0f / kTwoPi;
}  // namespace

Controller::Controller(float frequency, float ramp_time, float control_hz,
                       std::vector<ServoParams> servos)
    : omega_(kTwoPi * frequency),
      ramp_time_(ramp_time),
      dt_(control_hz > 0.0f ? 1.0f / control_hz : 0.0f),
      servos_(std::move(servos)) {
    const size_t n = servos_.size();
    theta_.assign(n, 0.0f);
    targets.assign(n, 0.0f);
    steps.assign(n, 0);
    wheel_speeds.assign(n, 0.0f);

    // A servo cannot follow a signal whose peak speed exceeds its own.
    for (ServoParams& p : servos_) {
        if (p.mode != ServoParams::Mode::Position) continue;
        const float peak = (p.amp1 + 2.0f * p.amp2) * omega_;
        const float k = std::min(1.0f, kServoSpeed / std::max(peak, 1e-12f));
        p.amp1 *= k;
        p.amp2 *= k;
    }
}

void Controller::tick(const float* loads_normalised) {
    const float ramp =
        ramp_time_ > 0.0f ? std::min(1.0f, time_ / ramp_time_) : 1.0f;
    const float h = dt_ / static_cast<float>(kSubsteps);

    for (size_t i = 0; i < servos_.size(); ++i) {
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

        const float n =
            loads_normalised
                ? std::clamp(p.sign * loads_normalised[i], -1.0f, 1.0f)
                : 0.0f;
        for (int k = 0; k < kSubsteps; ++k) {
            const float dtheta = omega_ - p.sigma * n * std::cos(theta_[i]);
            theta_[i] += h * dtheta;
        }
    }

    time_ += dt_;
}

}  // namespace darwinbot

// Portable DarwinBot control core: the same Tegotae CPG as
// darwinbot/controller.py, with no Arduino or external dependency.
//
// Position servo:
//   dtheta/dt = omega - sigma * N * cos(theta)          (5 substeps per tick)
//   N         = clip(sign * load_normalised / load_full_scale, -1, 1)
//   theta is wrapped to [0, 2pi) every tick
//   target    = ramp(t) * (offset + amp1 sin(theta + phase1)
//                                  + amp2 sin(2 theta + phase2))
// Wheel: speed = ramp(t) * speed * sign, phase and load ignored.
// Steering: command c in [-1, 1] scales position amplitudes by
// clip(1 + c * steer_gain, 0, 2), shifts position offsets by c * steer_shift
// and shifts wheel speeds by c * steer_shift (clipped to +/-5.45 rad/s).
// Steering is applied before the amplitude speed clip.
// Amplitudes are clipped so (amp1 + 2 amp2) * omega <= 5.45 rad/s.
#pragma once

#include <vector>

namespace darwinbot {

struct ServoParams {
    enum class Mode { Position, Wheel };

    Mode mode = Mode::Position;
    float amp1 = 0.0f;      // rad, first harmonic
    float phase1 = 0.0f;    // rad
    float offset = 0.0f;    // rad (wheel: signed speed, rad/s)
    float amp2 = 0.0f;      // rad, second harmonic
    float phase2 = 0.0f;    // rad
    float sigma = 0.0f;     // rad/s, load feedback gain (0 = open loop)
    float steer_gain = 0.0f;   // amplitude change at full steering command
    float steer_shift = 0.0f;  // rad, offset change at full command (wheel: rad/s)
    float sign = 1.0f;      // +1 or -1, mounting direction
    float zero = 2048.0f;   // servo counts at target 0 rad
    float min = 0.0f;       // rad, joint limit
    float max = 0.0f;       // rad, joint limit
    float speed = 0.0f;     // rad/s, wheel mode
};

class Controller {
public:
    static constexpr int kSubsteps = 5;       // 4 ms each at 50 Hz
    static constexpr float kServoSpeed = 5.45f;
    static constexpr float kLoadFullScale = 0.7f;   // control.json default

    Controller(float frequency, float ramp_time, float control_hz,
               std::vector<ServoParams> servos,
               float load_full_scale = kLoadFullScale);

    // One control period: targets from the current phases at t, then
    // kSubsteps explicit-Euler phase steps under the same loads.
    // loads_normalised: signed bus load / max torque (the servo's full
    // scale, not the simulation's). The core divides by load_full_scale (the
    // fraction of that torque that counts as N = 1) and applies the sign, per
    // docs/formato-controle.md.
    void tick(const float* loads_normalised);

    // Steering command in [-1, 1] (0 = straight, -1 left, +1 right). Applies
    // the steering genes and the speed clip; does not advance time.
    void set_command(float c);

    std::vector<float> targets;       // position: rad (clamped); wheel: rad/s
    std::vector<int> steps;           // position: servo counts, 0 on wheels
    std::vector<float> wheel_speeds;  // wheel: signed rad/s, 0 on joints
    const std::vector<float>& phases() const { return theta_; }

private:
    float omega_;
    float ramp_time_;
    float dt_;
    float load_full_scale_;
    float time_ = 0.0f;
    float command_ = 0.0f;
    std::vector<ServoParams> servos_;
    std::vector<ServoParams> base_;   // decoded values, before steering
    std::vector<float> theta_;
};

}  // namespace darwinbot

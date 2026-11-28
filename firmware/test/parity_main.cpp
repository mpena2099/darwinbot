// Parity harness: run the portable core on a recorded case and compare its
// targets with the Python controller.
//
// Case file (plain text, one record per line, blank lines and lines starting
// with '#' are ignored):
//
//   frequency ramp_time control_hz n_servos [load_full_scale [command [speed [servo_speed]]]]
//   <servo line>                                (n_servos times)
//   loads (n_servos) expected targets (n_servos) expected wheel speeds (n_servos)
//   expected phases (n_servos)                  (one line per tick)
//
// A line holding only "reset" calls Controller::reset() and is not a tick.
//
// A servo line is always eighteen numbers after the mode:
//   position amp1 phase1 offset amp2 phase2 sigma steer_gain steer_shift speed_gain speed_shift couple_to couple_k couple_lag sign zero min max speed
//   wheel    0    0      0      0    0      0     0    0          0           0         0           couple_to couple_k couple_lag sign 0    0   0   speed
//
// The values are the decoded genes before steering (amp1_base, amp2_base and
// offset_base in controller.py). couple_to is the layout index of the partner
// servo (-1 for none), couple_k its strength (rad/s) and couple_lag its phase
// offset (rad), as in format 5 of docs/formato-controle.md. Loads are bus loads
// over the servo maximum
// torque (signed, in [-1, 1]); the core divides them by load_full_scale
// (default 0.7). The optional command is the steering command in [-1, 1] and the optional
// speed the speed command in [0, 1] (1 if absent), the last the servo no-load
// speed in rad/s (4.71 if absent).
// Expected targets are radians for position servos and rad/s for wheels
// (unsigned). Expected wheel speeds are the signed rad/s a wheel servo is
// commanded, 0 for position servos. Expected phases are the controller's
// phases after the tick, compared modulo 2pi.
//
// Usage: parity_main case.txt
// Prints the maximum absolute error (targets, wheel speeds and phases) and
// exits 1 if it exceeds 1e-3.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../core/darwin_control.h"

namespace {

bool read_record(std::istringstream& in, std::vector<float>& out) {
    for (float& v : out) {
        if (!(in >> v)) return false;
    }
    return true;
}

// Smallest difference between two angles, in [-pi, pi].
float angle_error(float a, float b) {
    const float kPi = 3.14159265358979323846f;
    const float kTwoPi = 2.0f * kPi;
    float d = std::fmod(a - b, kTwoPi);
    if (d > kPi) d -= kTwoPi;
    if (d < -kPi) d += kTwoPi;
    return std::fabs(d);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s case.txt\n", argv[0]);
        return 2;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    std::vector<std::string> lines;
    for (std::string line; std::getline(file, line);) {
        const size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        lines.push_back(line);
    }

    float frequency = 0.0f, ramp_time = 0.0f, control_hz = 0.0f;
    float load_full_scale = darwinbot::Controller::kLoadFullScale;
    float command = 0.0f;
    float speed_cmd = 1.0f;
    float servo_speed = darwinbot::Controller::kServoSpeed;
    std::string clip_name = "command";
    int n_servos = 0;
    {
        std::istringstream header(lines.empty() ? "" : lines[0]);
        if (!(header >> frequency >> ramp_time >> control_hz >> n_servos) ||
            n_servos <= 0) {
            std::fprintf(stderr, "bad header\n");
            return 2;
        }
        header >> load_full_scale;      // optional, keeps the default if absent
        header >> command;              // optional, 0 (straight) if absent
        header >> speed_cmd;            // optional, 1 (full speed) if absent
        header >> servo_speed;          // optional, 4.71 if absent
        header >> clip_name;            // optional, "command" if absent
    }
    const darwinbot::Controller::ClipMode clip_mode =
        clip_name == "headroom" ? darwinbot::Controller::ClipMode::Headroom
                                 : darwinbot::Controller::ClipMode::Command;

    if (lines.size() < static_cast<size_t>(n_servos) + 1) {
        std::fprintf(stderr, "missing servo lines\n");
        return 2;
    }

    std::vector<darwinbot::ServoParams> servos(
        static_cast<size_t>(n_servos));
    for (int i = 0; i < n_servos; ++i) {
        std::istringstream in(lines[static_cast<size_t>(i) + 1]);
        std::string mode;
        darwinbot::ServoParams& p = servos[static_cast<size_t>(i)];
        if (!(in >> mode >> p.amp1 >> p.phase1 >> p.offset >> p.amp2 >>
              p.phase2 >> p.sigma >> p.steer_gain >> p.steer_shift >> p.speed_gain >>
              p.speed_shift >> p.couple_to >> p.couple_k >> p.couple_lag >>
              p.sign >> p.zero >> p.min >> p.max >> p.speed) ||
            (mode != "position" && mode != "wheel")) {
            std::fprintf(stderr, "bad servo line %d\n", i + 1);
            return 2;
        }
        p.mode = mode == "wheel" ? darwinbot::ServoParams::Mode::Wheel
                                 : darwinbot::ServoParams::Mode::Position;
    }

    darwinbot::Controller controller(frequency, ramp_time, control_hz,
                                     std::move(servos), load_full_scale,
                                     servo_speed, clip_mode);
    controller.set_command(command);
    controller.set_speed(speed_cmd);
    std::vector<float> loads(static_cast<size_t>(n_servos));
    std::vector<float> expected(static_cast<size_t>(n_servos));
    std::vector<float> expected_speed(static_cast<size_t>(n_servos));
    std::vector<float> expected_phase(static_cast<size_t>(n_servos));
    float max_error = 0.0f;
    int ticks = 0;

    for (size_t i = static_cast<size_t>(n_servos) + 1; i < lines.size(); ++i) {
        std::istringstream in(lines[i]);
        std::string word;
        if ((in >> word) && word == "reset") {
            controller.reset();
            continue;
        }
        in.clear();
        in.seekg(0);
        if (!read_record(in, loads) || !read_record(in, expected) ||
            !read_record(in, expected_speed) ||
            !read_record(in, expected_phase)) {
            std::fprintf(stderr, "bad tick line %zu\n", i + 1);
            return 2;
        }
        controller.tick(loads.data());
        for (int s = 0; s < n_servos; ++s) {
            const float error = std::fabs(
                controller.targets[static_cast<size_t>(s)] -
                expected[static_cast<size_t>(s)]);
            if (error > max_error) max_error = error;
            const float speed_error = std::fabs(
                controller.wheel_speeds[static_cast<size_t>(s)] -
                expected_speed[static_cast<size_t>(s)]);
            if (speed_error > max_error) max_error = speed_error;
            const float phase_error = angle_error(
                controller.phases()[static_cast<size_t>(s)],
                expected_phase[static_cast<size_t>(s)]);
            if (phase_error > max_error) max_error = phase_error;
        }
        ++ticks;
    }

    std::printf("ticks: %d, max abs error: %.9g\n", ticks,
                static_cast<double>(max_error));
    return max_error <= 1e-3f ? 0 : 1;
}

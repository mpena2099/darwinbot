// Parity harness: run the portable core on a recorded case and compare its
// targets with the Python controller.
//
// Case file (plain text, one record per line, blank lines and lines starting
// with '#' are ignored):
//
//   frequency ramp_time control_hz n_servos
//   <servo line>                                (n_servos times)
//   loads (n_servos) expected targets (n_servos)   (one line per tick)
//
// A servo line is always eleven numbers after the mode:
//   position amp1 phase1 offset amp2 phase2 sigma sign zero min max speed
//   wheel    0    0      0      0    0      0     1    0    0   0   speed
//
// Loads are normalised bus loads (load / max torque, signed, in [-1, 1]).
// Expected targets are radians for position servos and rad/s for wheels.
//
// Usage: parity_main case.txt
// Prints the maximum absolute error and exits 1 if it exceeds 1e-3 rad.

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
    int n_servos = 0;
    {
        std::istringstream header(lines.empty() ? "" : lines[0]);
        if (!(header >> frequency >> ramp_time >> control_hz >> n_servos) ||
            n_servos <= 0) {
            std::fprintf(stderr, "bad header\n");
            return 2;
        }
    }

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
              p.phase2 >> p.sigma >> p.sign >> p.zero >> p.min >> p.max >>
              p.speed) ||
            (mode != "position" && mode != "wheel")) {
            std::fprintf(stderr, "bad servo line %d\n", i + 1);
            return 2;
        }
        p.mode = mode == "wheel" ? darwinbot::ServoParams::Mode::Wheel
                                 : darwinbot::ServoParams::Mode::Position;
    }

    darwinbot::Controller controller(frequency, ramp_time, control_hz,
                                     std::move(servos));
    std::vector<float> loads(static_cast<size_t>(n_servos));
    std::vector<float> expected(static_cast<size_t>(n_servos));
    float max_error = 0.0f;
    int ticks = 0;

    for (size_t i = static_cast<size_t>(n_servos) + 1; i < lines.size(); ++i) {
        std::istringstream in(lines[i]);
        if (!read_record(in, loads) || !read_record(in, expected)) {
            std::fprintf(stderr, "bad tick line %zu\n", i + 1);
            return 2;
        }
        controller.tick(loads.data());
        for (int s = 0; s < n_servos; ++s) {
            const float error = std::fabs(
                controller.targets[static_cast<size_t>(s)] -
                expected[static_cast<size_t>(s)]);
            if (error > max_error) max_error = error;
        }
        ++ticks;
    }

    std::printf("ticks: %d, max abs error: %.9g rad\n", ticks,
                static_cast<double>(max_error));
    return max_error <= 1e-3f ? 0 : 1;
}

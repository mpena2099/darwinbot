#include "bus.h"

#include <cstring>

namespace darwinbot_esp32 {
namespace {

constexpr uint8_t kHeader = 0xFF;
constexpr uint8_t kBroadcast = 0xFE;
constexpr uint8_t kInstructionRead = 0x02;
constexpr uint8_t kInstructionWrite = 0x03;
constexpr uint8_t kInstructionSyncWrite = 0x83;
constexpr uint8_t kMaxPayload = 8;

HardwareSerial& bus = Serial2;

}  // namespace

uint8_t Bus::checksum(const uint8_t* p, size_t n) {
    uint8_t sum = 0;
    for (size_t i = 0; i < n; ++i) sum += p[i];
    return static_cast<uint8_t>(~sum);
}

void Bus::begin() {
    pinMode(kDirPin, OUTPUT);
    digitalWrite(kDirPin, LOW);   // listen while idle
    bus.begin(kBaud, SERIAL_8N1, kRxPin, kTxPin);
}

void Bus::send(const uint8_t* packet, size_t n) {
    while (bus.available()) bus.read();   // drop anything a slow answer left behind
    digitalWrite(kDirPin, HIGH);          // the adapter drives the bus from TX
    bus.write(packet, n);
    bus.flush();                          // the last stop bit is out before we listen
    digitalWrite(kDirPin, LOW);
}

bool Bus::get_byte(uint8_t& b) {
    const uint32_t start = micros();
    while (micros() - start < kByteTimeoutUs) {
        if (bus.available()) {
            b = static_cast<uint8_t>(bus.read());
            return true;
        }
    }
    return false;
}

bool Bus::read_answer(uint8_t id, uint8_t* params, int& n) {
    uint8_t b;
    if (!get_byte(b) || b != kHeader) return false;
    if (!get_byte(b) || b != kHeader) return false;
    uint8_t head[3];   // id, length, error
    for (uint8_t& h : head) {
        if (!get_byte(h)) return false;
    }
    if (head[0] != id) return false;
    const uint8_t length = head[1];
    if (length < 2 || length - 2 > kMaxPayload) return false;
    const int count = length - 2;
    uint8_t sum = head[0] + head[1] + head[2];
    for (int i = 0; i < count; ++i) {
        if (!get_byte(params[i])) return false;
        sum += params[i];
    }
    if (!get_byte(b) || static_cast<uint8_t>(~sum) != b) return false;
    n = count;
    return true;
}

bool Bus::write_reg(uint8_t id, uint8_t reg, uint8_t value) {
    // A status return level of 1 answers reads only, so a write needs no answer.
    const uint8_t packet[] = {kHeader, kHeader, id, 4, kInstructionWrite, reg, value, 0};
    uint8_t p[sizeof(packet)];
    std::memcpy(p, packet, sizeof(packet));
    p[7] = checksum(p + 2, 5);
    send(p, sizeof(p));
    return true;
}

bool Bus::read_regs(uint8_t id, uint8_t reg, uint8_t* out, uint8_t len) {
    const uint8_t packet[] = {kHeader, kHeader, id, 4, kInstructionRead, reg, len, 0};
    uint8_t p[sizeof(packet)];
    std::memcpy(p, packet, sizeof(packet));
    p[7] = checksum(p + 2, 5);
    send(p, sizeof(p));
    uint8_t params[kMaxPayload];
    int n = 0;
    if (!read_answer(id, params, n) || n < len) return false;
    std::memcpy(out, params, len);
    return true;
}

void Bus::set_mode(uint8_t id, bool wheel) {
    write_reg(id, kRegStatusReturn, 1);                  // answer reads only
    write_reg(id, kRegTorqueEnable, 0);                  // free before the mode
    write_reg(id, kRegMode, wheel ? kModeWheel : kModePosition);
    if (!wheel) {                                        // move speed limit
        const uint16_t speed = kPositionGoalSpeed;
        write_regs(kRegGoalSpeed, &id, &speed, 1);
    }
    write_reg(id, kRegTorqueEnable, 1);
}

bool Bus::read_load(uint8_t id, float& load) {
    uint8_t raw[2];
    if (!read_regs(id, kRegPresentLoad, raw, 2)) return false;
    const uint16_t word = static_cast<uint16_t>(raw[0] | (raw[1] << 8));
    const float magnitude = static_cast<float>(word & 0x03FF) / 1000.0f;  // 100 % = 1
    load = (word & 0x0400) ? -magnitude : magnitude;                      // bit 10
    return true;
}

bool Bus::read_voltage(uint8_t id, float& volts) {
    uint8_t raw = 0;
    if (!read_regs(id, kRegPresentVoltage, &raw, 1)) return false;
    volts = static_cast<float>(raw) / 10.0f;
    return true;
}

void Bus::write_regs(uint8_t reg, const uint8_t* ids, const uint16_t* values, int n) {
    if (n <= 0) return;
    uint8_t p[8 + 3 * kMaxServos];   // header, id, length, instruction, register, size, data, checksum
    size_t k = 0;
    p[k++] = kHeader;
    p[k++] = kHeader;
    p[k++] = kBroadcast;
    p[k++] = static_cast<uint8_t>(4 + 3 * n);   // instruction, register, size, data, checksum
    p[k++] = kInstructionSyncWrite;
    p[k++] = reg;
    p[k++] = 2;
    for (int i = 0; i < n; ++i) {
        p[k++] = ids[i];
        p[k++] = static_cast<uint8_t>(values[i] & 0xFF);
        p[k++] = static_cast<uint8_t>(values[i] >> 8);
    }
    p[k] = checksum(p + 2, k - 2);
    send(p, k + 1);
}

}  // namespace darwinbot_esp32

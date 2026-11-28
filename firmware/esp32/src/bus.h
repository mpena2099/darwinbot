// Feetech STS3215 bus over one half-duplex UART at 1 Mbps.
//
// Register addresses from the STS3215 memory table (Feetech STS/SMS/SCS
// communication protocol), the ones this firmware touches:
//   0x07 return delay time      0x08 status return level (1 = answer reads only)
//   0x21 mode                   (0 = position servo, 1 = wheel)
//   0x28 torque enable          (0 = free, 1 = driven)
//   0x2A goal position          (2 bytes, 0..4095)
//   0x2E goal speed             (2 bytes, steps/s; position: move speed limit,
//                                wheel: sign in bit 15, magnitude in 15 bits)
//   0x30 torque limit           (2 bytes, 0..1000 = 0..100 % of the maximum torque)
//   0x37 EEPROM lock            (0 = EEPROM writable, 1 = locked)
//   0x38 present position       (2 bytes, 0..4095)
//   0x3C present load           (2 bytes, sign in bit 10, 0..1000 = 0..100 %
//                                of the maximum torque)
//   0x3E present voltage        (1 byte, 0.1 V)
// Protocol: 0xFF 0xFF ID LEN INSTRUCTION PARAMS CHECKSUM, checksum = ~(ID + LEN
// + INSTRUCTION + PARAMS). In a status packet the byte after LEN is the error
// byte (bit 0 voltage, 1 angle sensor, 2 overheat, 3 overcurrent, 5 overload).
// Broadcast id 0xFE with instruction 0x83 is SYNC WRITE.
#pragma once

#include <Arduino.h>

#include <cstddef>
#include <cstdint>

namespace darwinbot_esp32 {

constexpr int kMaxServos = 16;             // morphology.MAX_SERVOS

constexpr int kTxPin = 17;                 // UART2 TX, to the half-duplex adapter
constexpr int kRxPin = 16;                 // UART2 RX, from the half-duplex adapter
constexpr int kDirPin = 4;                 // adapter OE: HIGH drives the bus, LOW listens
constexpr uint32_t kBaud = 1000000;        // STS3215 factory baud rate
// DARWIN_BUS_ECHO=1 (default): the adapter loops every transmitted byte back to
// RX (single-wire and 74HCT126 wirings do), so the firmware discards exactly the
// bytes it sent before it reads an answer. Build with -DDARWIN_BUS_ECHO=0 for an
// adapter that keeps RX silent while transmitting.
#ifndef DARWIN_BUS_ECHO
#define DARWIN_BUS_ECHO 1
#endif

constexpr uint32_t kByteTimeoutUs = 1000;  // per byte of the servo's answer

constexpr uint8_t kRegStatusReturn = 0x08;
constexpr uint8_t kRegMode = 0x21;
constexpr uint8_t kRegTorqueEnable = 0x28;
constexpr uint8_t kRegGoalPosition = 0x2A;
constexpr uint8_t kRegTorqueLimit = 0x30;
constexpr uint8_t kRegLock = 0x37;
constexpr uint8_t kRegPresentPosition = 0x38;
constexpr uint8_t kRegGoalSpeed = 0x2E;
constexpr uint8_t kRegPresentLoad = 0x3C;
constexpr uint8_t kRegPresentVoltage = 0x3E;

constexpr uint8_t kModePosition = 0;
constexpr uint8_t kModeWheel = 1;

constexpr float kStepsPerRad = 4096.0f / 6.283185307179586f;

// Status error byte bits that mean the servo protected itself: overheat,
// overcurrent, overload.
constexpr uint8_t kErrorMask = 0x04 | 0x08 | 0x20;

class Bus {
public:
    void begin();

    // Status return level and mode, written to EEPROM only when they differ
    // (torque off first, EEPROM unlocked around the write). False if the servo
    // does not answer.
    bool set_mode(uint8_t id, bool wheel);

    // Torque limit, as a fraction of the maximum torque in (0, 1].
    void set_torque_limit(uint8_t id, float fraction);

    // Puts a servo to work without a jump: a position servo reads its present
    // position (returned in steps) and takes it as its goal, with the move speed
    // limit at speed_steps; a wheel gets speed 0. Then torque on. False (torque
    // stays off) if the position cannot be read.
    bool arm(uint8_t id, bool wheel, uint16_t speed_steps, uint16_t& position);

    void torque(uint8_t id, bool on);

    // Signed bus load over the maximum torque, in [-1, 1]. False on a timeout.
    bool read_load(uint8_t id, float& load);

    // Error byte of the last status packet (see kErrorMask).
    uint8_t last_error() const { return last_error_; }

    // Supply voltage in volts (one servo answers for the battery).
    bool read_voltage(uint8_t id, float& volts);

    // SYNC WRITE of one 2-byte register to one group of servos. Both are small
    // enough that the whole group fits in one packet.
    void write_regs(uint8_t reg, const uint8_t* ids, const uint16_t* values, int n);

private:
    void send(const uint8_t* packet, size_t n);
    bool write_reg(uint8_t id, uint8_t reg, uint8_t value);
    bool write_eeprom(uint8_t id, uint8_t reg, uint8_t value);
    bool read_regs(uint8_t id, uint8_t reg, uint8_t* out, uint8_t len);
    bool read_answer(uint8_t id, uint8_t* params, uint8_t expect);
    bool get_byte(uint8_t& b);
    uint8_t last_error_ = 0;
    static uint8_t checksum(const uint8_t* p, size_t n);
};

}  // namespace darwinbot_esp32

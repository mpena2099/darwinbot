// Wi-Fi, the page in LittleFS and the 10 Hz WebSocket telemetry.
#pragma once

#include <cstdint>

namespace darwinbot_esp32 {

struct NetConfig {
    bool station = false;      // false: the robot is its own access point
    char ap[24] = "";          // darwinbot-XXXX, from the last MAC bytes
    char ap_password[65] = "";   // WPA2 password of the access point
    char ssid[33] = "";
    char password[65] = "";
    char agent_ip[40] = "";
    uint16_t agent_port = 8888;
};

// Reads /wifi.json {ssid, password, ros_agent_ip, ros_agent_port, ap_password}
// when present and joins that network; otherwise starts the WPA2 access point
// (password "darwinbot" unless ap_password is set). Prints the address.
void net_begin();

// Must follow net_begin(): HTTP page on port 80, WebSocket on port 81.
void web_begin();

// Serves the page, the WebSocket and the 10 Hz telemetry. Call from loop().
void web_loop();

const NetConfig& net_config();

}  // namespace darwinbot_esp32

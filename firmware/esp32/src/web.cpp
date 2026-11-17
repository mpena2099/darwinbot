#include "web.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <LittleFS.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <WiFi.h>

#include <cstdio>
#include <cstring>

#include "command.h"
#include "telemetry.h"

namespace darwinbot_esp32 {
namespace {

NetConfig g_net;
WebServer g_http(80);
WebSocketsServer g_ws(81);
uint32_t g_last_push = 0;

void on_ws(uint8_t client, WStype_t type, uint8_t* payload, size_t len) {
    if (type != WStype_TEXT) return;
    char text[32];
    if (len >= sizeof(text)) len = sizeof(text) - 1;
    std::memcpy(text, payload, len);
    text[len] = '\0';

    float value = 0.0f;
    if (std::strcmp(text, "stop") == 0) {
        cmd::stop();
    } else if (sscanf(text, "cmd %f", &value) == 1) {
        cmd::set_command(value);
    } else if (sscanf(text, "speed %f", &value) == 1) {
        cmd::set_speed(value);
    }
}

void push_telemetry() {
    Telemetry t;
    telemetry_copy(t);
    char out[512];
    int k = snprintf(out, sizeof(out), "{\"c\":%.2f,\"s\":%.2f,\"v\":%.1f,\"load\":[",
                     t.command, t.speed, t.volts);
    for (int i = 0; i < t.n && k < static_cast<int>(sizeof(out)) - 32; ++i) {
        k += snprintf(out + k, sizeof(out) - k, "%s%.2f", i ? "," : "", t.load[i]);
    }
    k += snprintf(out + k, sizeof(out) - k, "],\"phase\":[");
    for (int i = 0; i < t.n && k < static_cast<int>(sizeof(out)) - 16; ++i) {
        k += snprintf(out + k, sizeof(out) - k, "%s%.2f", i ? "," : "", t.phase[i]);
    }
    snprintf(out + k, sizeof(out) - k, "]}");
    g_ws.broadcastTXT(out);
}

}  // namespace

const NetConfig& net_config() { return g_net; }

void net_begin() {
    uint8_t mac[6];
    WiFi.macAddress(mac);
    snprintf(g_net.ap, sizeof(g_net.ap), "darwinbot-%02X%02X", mac[4], mac[5]);

    File f = LittleFS.open("/wifi.json", "r");
    if (f) {
        JsonDocument doc;
        const bool ok = !deserializeJson(doc, f);
        f.close();
        if (ok && doc["ssid"].is<const char*>()) {
            g_net.station = true;
            strlcpy(g_net.ssid, doc["ssid"] | "", sizeof(g_net.ssid));
            strlcpy(g_net.password, doc["password"] | "", sizeof(g_net.password));
            strlcpy(g_net.agent_ip, doc["ros_agent_ip"] | "", sizeof(g_net.agent_ip));
            g_net.agent_port = doc["ros_agent_port"] | 8888;
        } else {
            Serial.println("wifi.json is not {ssid, password, ros_agent_ip, ros_agent_port}: access point");
        }
    }

    if (g_net.station) {
        WiFi.mode(WIFI_STA);
        WiFi.begin(g_net.ssid, g_net.password);
        Serial.printf("Wi-Fi: joining '%s'", g_net.ssid);
        for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; ++i) {
            delay(250);
            Serial.print('.');
        }
        Serial.println();
        if (WiFi.status() == WL_CONNECTED) {
            Serial.printf("Wi-Fi: http://%s/\n", WiFi.localIP().toString().c_str());
        } else {
            Serial.println("Wi-Fi: not connected yet, still trying in the background");
        }
    } else {
        WiFi.mode(WIFI_AP);
        WiFi.softAP(g_net.ap);
        Serial.printf("Wi-Fi: access point '%s', http://%s/\n", g_net.ap,
                      WiFi.softAPIP().toString().c_str());
    }
}

void web_begin() {
    g_http.on("/", HTTP_GET, [] {
        File page = LittleFS.open("/index.html", "r");
        if (!page) {
            g_http.send(500, "text/plain", "index.html missing: run 'pio run -t uploadfs'");
            return;
        }
        g_http.streamFile(page, "text/html");
        page.close();
    });
    g_http.onNotFound([] { g_http.send(404, "text/plain", "not found"); });
    g_http.begin();

    g_ws.begin();
    g_ws.onEvent(on_ws);
}

void web_loop() {
    g_http.handleClient();
    g_ws.loop();
    if (millis() - g_last_push >= 100) {   // 10 Hz
        g_last_push = millis();
        push_telemetry();
    }
}

}  // namespace darwinbot_esp32

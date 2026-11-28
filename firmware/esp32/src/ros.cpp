#include "ros.h"

#ifdef DARWIN_ROS

#include <Arduino.h>
#include <geometry_msgs/msg/twist.h>
#include <micro_ros_platformio.h>
#include <rcl/rcl.h>
#include <rclc/executor.h>
#include <rclc/rclc.h>
#include <rmw_microros/rmw_microros.h>
#include <sensor_msgs/msg/joint_state.h>

#include <cmath>
#include <cstring>

#include "bus.h"
#include "command.h"
#include "telemetry.h"
#include "web.h"

namespace darwinbot_esp32 {
namespace {

// Twist scales: the robot's own top speed and yaw rate, from the bench.
constexpr float kMaxSpeed = 1.0f;    // m/s at speed 1
constexpr float kMaxYawRate = 1.0f;  // rad/s at full steering command
constexpr uint32_t kPublishMs = 100;

Config g_cfg;
rclc_support_t g_support;
rcl_node_t g_node;
rcl_subscription_t g_sub;
rcl_publisher_t g_pub;
geometry_msgs__msg__Twist g_twist;
sensor_msgs__msg__JointState g_state;
rosidl_runtime_c__String g_names[kMaxServos];
double g_position[kMaxServos];
double g_effort[kMaxServos];

float clampf(float v, float lo, float hi) {
    if (!std::isfinite(v)) return 0.0f;
    return v < lo ? lo : (v > hi ? hi : v);
}

// angular.z positive turns left, which is a negative steering command.
void on_cmd_vel(const void* msg) {
    const auto* twist = static_cast<const geometry_msgs__msg__Twist*>(msg);
    const float z = static_cast<float>(twist->angular.z);
    const float x = static_cast<float>(twist->linear.x);
    if (!std::isfinite(z) || !std::isfinite(x)) return;
    const float steer = -clampf(z / kMaxYawRate, -1.0f, 1.0f);
    const float speed = clampf(x / kMaxSpeed, 0.0f, 1.0f);
    cmd::set(steer, speed);   // stops only when both are 0
}

void joint_state_init() {
    g_state = sensor_msgs__msg__JointState();
    for (int i = 0; i < g_cfg.n; ++i) {
        g_names[i].data = g_cfg.names[i];
        g_names[i].size = std::strlen(g_cfg.names[i]);
        g_names[i].capacity = sizeof(g_cfg.names[i]);
    }
    g_state.name.data = g_names;
    g_state.name.size = g_cfg.n;
    g_state.name.capacity = kMaxServos;
    g_state.position.data = g_position;
    g_state.position.size = g_cfg.n;
    g_state.position.capacity = kMaxServos;
    g_state.effort.data = g_effort;
    g_state.effort.size = g_cfg.n;
    g_state.effort.capacity = kMaxServos;
}

void publish_joint_state() {
    Telemetry t;
    telemetry_copy(t);
    for (int i = 0; i < t.n && i < g_cfg.n; ++i) {
        g_position[i] = t.target[i];
        g_effort[i] = t.load[i];
    }
    if (rcl_publish(&g_pub, &g_state, nullptr) != RCL_RET_OK) rcl_reset_error();
}

// One agent session: create the entities, run until the agent stops answering,
// then tear them down so the next session starts clean.
void session() {
    rcl_allocator_t alloc = rcl_get_default_allocator();
    if (rclc_support_init(&g_support, 0, nullptr, &alloc) != RCL_RET_OK) return;
    if (rclc_node_init_default(&g_node, "darwinbot", "", &g_support) != RCL_RET_OK) {
        rclc_support_fini(&g_support);
        return;
    }
    const rosidl_message_type_support_t* twist =
        ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist);
    const rosidl_message_type_support_t* joint_state =
        ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState);
    if (rclc_subscription_init_best_effort(&g_sub, &g_node, twist, "cmd_vel") != RCL_RET_OK ||
        rclc_publisher_init_default(&g_pub, &g_node, joint_state,
                                    "darwinbot/joint_states") != RCL_RET_OK) {
        (void)rcl_node_fini(&g_node);
        rclc_support_fini(&g_support);
        return;
    }

    rclc_executor_t executor;
    if (rclc_executor_init(&executor, &g_support.context, 1, &alloc) != RCL_RET_OK) {
        (void)rcl_subscription_fini(&g_sub, &g_node);
        (void)rcl_publisher_fini(&g_pub, &g_node);
        (void)rcl_node_fini(&g_node);
        rclc_support_fini(&g_support);
        return;
    }
    g_twist = geometry_msgs__msg__Twist();
    joint_state_init();
    if (rclc_executor_add_subscription(&executor, &g_sub, &g_twist, &on_cmd_vel, ON_NEW_DATA) !=
        RCL_RET_OK) {
        Serial.println("ROS: could not add the cmd_vel subscription");
    }

    uint32_t last = millis();
    while (rmw_uros_ping_agent(200, 1) == RMW_RET_OK) {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(20));
        if (millis() - last >= kPublishMs) {
            last = millis();
            publish_joint_state();
        }
    }

    rclc_executor_fini(&executor);
    (void)rcl_subscription_fini(&g_sub, &g_node);
    (void)rcl_publisher_fini(&g_pub, &g_node);
    (void)rcl_node_fini(&g_node);
    rclc_support_fini(&g_support);
}

void task(void*) {
    const NetConfig& net = net_config();
    IPAddress agent;
    if (!agent.fromString(net.agent_ip)) {
        Serial.printf("ROS: ros_agent_ip '%s' in /wifi.json is not an address, node not started\n",
                      net.agent_ip);
        vTaskDelete(nullptr);
    }
    // The transport API takes mutable strings and only reads them. It joins the
    // network again (already joined by net_begin) and blocks until connected.
    set_microros_wifi_transports(const_cast<char*>(net.ssid), const_cast<char*>(net.password),
                                 agent, net.agent_port);
    vTaskDelay(pdMS_TO_TICKS(500));
    for (;;) {
        if (rmw_uros_ping_agent(1000, 1) == RMW_RET_OK) {
            session();
        }
        vTaskDelay(pdMS_TO_TICKS(1000));   // no agent yet: try again, move nothing
    }
}

}  // namespace

void ros_begin(const Config& cfg) {
    g_cfg = cfg;
    xTaskCreatePinnedToCore(task, "ros", 16384, nullptr, 2, nullptr, 0);
}

}  // namespace darwinbot_esp32

#endif  // DARWIN_ROS

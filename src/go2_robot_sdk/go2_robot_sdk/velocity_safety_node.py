import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger


class VelocitySafetyNode(Node):
    """Gate all software velocity commands before they reach the robot driver."""

    def __init__(self):
        super().__init__("velocity_safety_node")
        self.declare_parameter("input_topic", "/cmd_vel_command")
        self.declare_parameter("output_topic", "/robot0/cmd_vel_out")
        self.declare_parameter("debug_topic", "/cmd_vel_safe_debug")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("motion_enabled", False)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("max_forward_speed", 0.3)
        self.declare_parameter("max_lateral_speed", 0.2)
        self.declare_parameter("max_yaw_rate", 0.4)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.motion_enabled = bool(self.get_parameter("motion_enabled").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.max_forward_speed = float(self.get_parameter("max_forward_speed").value)
        self.max_lateral_speed = float(self.get_parameter("max_lateral_speed").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)

        self.emergency_stop_input = False
        self.emergency_stop_latched = False
        self.last_command_time = None
        self.last_safe_command = Twist()
        self.hardware_command_active = False

        self.output_publisher = self.create_publisher(Twist, self.output_topic, 10)
        self.debug_publisher = self.create_publisher(Twist, self.debug_topic, 10)
        self.create_subscription(Twist, self.input_topic, self.command_cb, 10)
        self.create_subscription(
            Bool, "/software_emergency_stop", self.emergency_stop_cb, 10
        )
        self.create_service(SetBool, "~/enable_motion", self.enable_motion_cb)
        self.create_service(Trigger, "~/reset_emergency_stop", self.reset_stop_cb)
        self.timer = self.create_timer(0.05, self.watchdog_cb)

        self.publish_stop()
        self.get_logger().info(
            f"Velocity safety gate started (dry_run={self.dry_run}, "
            f"motion_enabled={self.motion_enabled})"
        )

    @staticmethod
    def finite_command(msg):
        values = (
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        )
        return all(math.isfinite(value) for value in values)

    def command_cb(self, msg):
        self.last_command_time = time.monotonic()
        if not self.finite_command(msg):
            self.get_logger().error("Rejected velocity command containing NaN or infinity")
            self.publish_stop()
            return

        safe = Twist()
        safe.linear.x = max(
            -self.max_forward_speed, min(self.max_forward_speed, msg.linear.x)
        )
        safe.linear.y = max(
            -self.max_lateral_speed, min(self.max_lateral_speed, msg.linear.y)
        )
        safe.angular.z = max(
            -self.max_yaw_rate, min(self.max_yaw_rate, msg.angular.z)
        )
        self.last_safe_command = safe
        self.publish_current_command()

    def emergency_stop_cb(self, msg):
        self.emergency_stop_input = bool(msg.data)
        if self.emergency_stop_input:
            self.emergency_stop_latched = True
            self.motion_enabled = False
            self.publish_stop()
            self.get_logger().error("Software emergency stop latched")

    def enable_motion_cb(self, request, response):
        if request.data and self.emergency_stop_latched:
            response.success = False
            response.message = "Reset the emergency-stop latch before enabling motion"
            return response
        self.motion_enabled = bool(request.data)
        if not self.motion_enabled:
            self.publish_stop()
        response.success = True
        response.message = f"motion_enabled={self.motion_enabled}"
        return response

    def reset_stop_cb(self, _request, response):
        if self.emergency_stop_input:
            response.success = False
            response.message = "Emergency-stop input is still active"
            return response
        self.emergency_stop_latched = False
        self.motion_enabled = False
        self.publish_stop()
        response.success = True
        response.message = "Emergency stop reset; motion remains disabled"
        return response

    def watchdog_cb(self):
        stale = (
            self.last_command_time is None
            or time.monotonic() - self.last_command_time > self.command_timeout
        )
        if stale or not self.motion_enabled or self.emergency_stop_latched:
            self.publish_stop()

    def publish_current_command(self):
        if not self.motion_enabled or self.emergency_stop_latched:
            self.publish_stop()
            return
        self.debug_publisher.publish(self.last_safe_command)
        if not self.dry_run:
            self.output_publisher.publish(self.last_safe_command)
        self.hardware_command_active = any(
            abs(value) > 1e-6
            for value in (
                self.last_safe_command.linear.x,
                self.last_safe_command.linear.y,
                self.last_safe_command.angular.z,
            )
        )

    def publish_stop(self):
        stop = Twist()
        self.debug_publisher.publish(stop)
        if not self.dry_run and self.hardware_command_active:
            self.output_publisher.publish(stop)
        self.hardware_command_active = False

    def destroy_node(self):
        for _ in range(3):
            self.publish_stop()
        return super().destroy_node()


def main():
    rclpy.init()
    node = VelocitySafetyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

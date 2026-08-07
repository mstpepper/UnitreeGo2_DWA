import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class VelocityTestPublisher(Node):
    """Publish one short, parameterized command for isolated pipeline testing."""

    def __init__(self):
        super().__init__("velocity_test_publisher")
        self.declare_parameter("output_topic", "/cmd_vel_command")
        self.declare_parameter("linear_x", 0.0)
        self.declare_parameter("linear_y", 0.0)
        self.declare_parameter("angular_z", 0.0)
        self.declare_parameter("duration", 1.0)
        self.declare_parameter("publish_rate", 20.0)

        self.publisher = self.create_publisher(
            Twist, self.get_parameter("output_topic").value, 10
        )
        self.command = Twist()
        self.command.linear.x = float(self.get_parameter("linear_x").value)
        self.command.linear.y = float(self.get_parameter("linear_y").value)
        self.command.angular.z = float(self.get_parameter("angular_z").value)
        self.duration = max(0.0, float(self.get_parameter("duration").value))
        rate = max(1.0, float(self.get_parameter("publish_rate").value))
        self.started = time.monotonic()
        self.finished = False
        self.timer = self.create_timer(1.0 / rate, self.timer_cb)

    def timer_cb(self):
        if time.monotonic() - self.started < self.duration:
            self.publisher.publish(self.command)
            return
        self.publisher.publish(Twist())
        self.finished = True
        self.get_logger().info("Velocity test complete; published stop command")
        self.timer.cancel()


def main():
    rclpy.init()
    node = VelocityTestPublisher()
    while rclpy.ok() and not node.finished:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

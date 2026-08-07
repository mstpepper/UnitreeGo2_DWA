import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from go2_interfaces.msg import Obstacle2DArray
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32, String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener

from .dwa_planner import DWAPlanner


class DWAControllerNode(Node):
    """ROS wrapper around DWAPlanner with independently testable inputs/output."""

    def __init__(self):
        super().__init__("dwa_controller_node")
        self.declare_parameter("bev_topic", "/bev/obstacles")
        self.declare_parameter("odom_topic", "/robot0/odom")
        self.declare_parameter("goal_topic", "/dwa/goal")
        self.declare_parameter("output_topic", "/cmd_vel_command")
        self.declare_parameter("preview_topic", "/dwa/cmd_vel_preview")
        self.declare_parameter("path_topic", "/dwa/predicted_path")
        self.declare_parameter("status_topic", "/dwa/status")
        self.declare_parameter("clearance_topic", "/dwa/minimum_clearance")
        self.declare_parameter("publish_motion_command", False)
        self.declare_parameter("control_rate", 5.0)
        self.declare_parameter("input_timeout", 0.5)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("heading_cost_gain", 0.15)
        self.declare_parameter("speed_cost_gain", 1.0)
        self.declare_parameter("obstacle_cost_gain", 1.0)
        self.declare_parameter("robot_stuck_flag_cons", 0.05)
        self.declare_parameter("stuck_turn_steps", 200)
        self.declare_parameter("stuck_detection_steps", 4)
        self.declare_parameter("recovery_turn_angle", 0.7)
        self.declare_parameter("recovery_forward_speed", 0.08)
        self.declare_parameter("recovery_bypass_distance", 1.2)

        self.planner = DWAPlanner()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_odom = None
        self.latest_goal = None
        self.last_obstacle_time = None
        self.last_odom_time = None
        self.publish_motion_command = bool(
            self.get_parameter("publish_motion_command").value
        )
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.input_timeout = float(self.get_parameter("input_timeout").value)
        self.planner_costs = {
            "to_goal_cost_gain": float(
                self.get_parameter("heading_cost_gain").value
            ),
            "speed_cost_gain": float(
                self.get_parameter("speed_cost_gain").value
            ),
            "obstacle_cost_gain": float(
                self.get_parameter("obstacle_cost_gain").value
            ),
            "robot_stuck_flag_cons": float(
                self.get_parameter("robot_stuck_flag_cons").value
            ),
            "stuck_turn_steps": float(
                self.get_parameter("stuck_turn_steps").value
            ),
            "stuck_detection_steps": float(
                self.get_parameter("stuck_detection_steps").value
            ),
            "recovery_turn_angle": float(
                self.get_parameter("recovery_turn_angle").value
            ),
            "recovery_forward_speed": float(
                self.get_parameter("recovery_forward_speed").value
            ),
            "recovery_bypass_distance": float(
                self.get_parameter("recovery_bypass_distance").value
            ),
        }

        self.motion_publisher = None
        if self.publish_motion_command:
            self.motion_publisher = self.create_publisher(
                Twist, self.get_parameter("output_topic").value, 10
            )
        self.preview_publisher = self.create_publisher(
            Twist, self.get_parameter("preview_topic").value, 10
        )
        self.path_publisher = self.create_publisher(
            Path, self.get_parameter("path_topic").value, 10
        )
        self.status_publisher = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10
        )
        self.clearance_publisher = self.create_publisher(
            Float32, self.get_parameter("clearance_topic").value, 10
        )
        self.latency_publisher = self.create_publisher(
            Float32, "/metrics/dwa_latency_ms", 10
        )
        self.create_subscription(
            Obstacle2DArray,
            self.get_parameter("bev_topic").value,
            self.bev_cb,
            10,
        )
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self.odom_cb, 10
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("goal_topic").value, self.goal_cb, 10
        )
        rate = max(1.0, float(self.get_parameter("control_rate").value))
        self.timer = self.create_timer(1.0 / rate, self.control_cb)
        self.publish_status("WAITING")
        self.get_logger().info(
            "DWA obstacle_source=bev, "
            f"publish_motion_command={self.publish_motion_command}"
        )

    @staticmethod
    def yaw_from_quaternion(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def odom_cb(self, msg):
        if msg.header.frame_id != "odom":
            self.latest_odom = None
            self.last_odom_time = None
            self.get_logger().warning(
                f"Rejected odometry in unsupported frame {msg.header.frame_id}; "
                "expected odom"
            )
            return
        self.latest_odom = msg
        self.last_odom_time = time.monotonic()

    def goal_cb(self, msg):
        source_frame = msg.header.frame_id
        if not source_frame:
            self.latest_goal = None
            self.get_logger().warning("Rejected DWA goal with an empty frame_id")
            return
        if source_frame == "odom":
            self.latest_goal = msg
            return
        point = PointStamped()
        point.header = msg.header
        point.point = msg.pose.position
        try:
            transform = self.tf_buffer.lookup_transform(
                "odom",
                source_frame,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.05),
            )
            transformed = do_transform_point(point, transform)
            goal = PoseStamped()
            goal.header = transformed.header
            goal.pose.position = transformed.point
            goal.pose.orientation.w = 1.0
            self.latest_goal = goal
        except TransformException as exc:
            self.latest_goal = None
            self.get_logger().warning(
                f"Cannot transform DWA goal from {source_frame} to odom: {exc}"
            )

    def bev_cb(self, msg):
        if msg.header.frame_id != "odom":
            self.last_obstacle_time = None
            self.get_logger().warning(
                f"Rejected BEV obstacles in frame {msg.header.frame_id!r}; "
                "expected 'odom'"
            )
            return
        self.planner.clear_obstacles()
        for obstacle in msg.obstacles:
            if (
                math.isfinite(obstacle.x)
                and math.isfinite(obstacle.y)
                and math.isfinite(obstacle.radius)
                and obstacle.radius > 0.0
            ):
                self.planner.add_obstacles(
                    obstacle.x, obstacle.y, obstacle.radius
                )
        self.last_obstacle_time = time.monotonic()

    def publish_status(self, status):
        message = String()
        message.data = status
        self.status_publisher.publish(message)

    def publish_command(self, command):
        self.preview_publisher.publish(command)
        if self.motion_publisher is not None:
            self.motion_publisher.publish(command)

    def publish_path(self, trajectory):
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "odom"
        for state in trajectory:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            pose.pose.orientation.z = math.sin(float(state[2]) / 2.0)
            pose.pose.orientation.w = math.cos(float(state[2]) / 2.0)
            message.poses.append(pose)
        self.path_publisher.publish(message)

    def control_cb(self):
        now = time.monotonic()
        if (
            self.latest_odom is None
            or self.latest_goal is None
            or self.last_obstacle_time is None
            or self.last_odom_time is None
            or now - self.last_obstacle_time > self.input_timeout
            or now - self.last_odom_time > self.input_timeout
        ):
            self.publish_command(Twist())
            self.publish_path([])
            inputs_missing = (
                self.latest_odom is None
                or self.latest_goal is None
                or self.last_obstacle_time is None
                or self.last_odom_time is None
            )
            self.publish_status("WAITING" if inputs_missing else "STALE")
            return

        pose = self.latest_odom.pose.pose
        goal = self.latest_goal.pose.position
        distance_to_goal = math.hypot(
            pose.position.x - goal.x, pose.position.y - goal.y
        )
        if distance_to_goal <= self.goal_tolerance:
            self.publish_command(Twist())
            self.publish_path([])
            self.publish_status("GOAL_REACHED")
            return

        planning_start = time.perf_counter()
        action = self.planner.get_robot_next_action(
            {
                "max_linear_speed": 0.3,
                "min_linear_speed": 0.0,
                "max_angular_speed": 0.4,
                "max_linear_acceleration": 0.3,
                "max_angular_acceleration": 0.8,
                "v_resolution": 0.05,
                "yaw_rate_resolution": 0.05,
                "robot_length": 0.7,
                "robot_width": 0.35,
                "safety_margin": 0.1,
                "predict_time": 3.0,
            },
            {
                "x": pose.position.x,
                "y": pose.position.y,
                "yaw": self.yaw_from_quaternion(pose.orientation),
                "linear_velocity": self.latest_odom.twist.twist.linear.x,
                "angular_velocity": self.latest_odom.twist.twist.angular.z,
            },
            {"x": goal.x, "y": goal.y},
            simulation_specs=self.planner_costs,
            dt=0.2,
        )
        planning_latency = Float32()
        planning_latency.data = float(
            (time.perf_counter() - planning_start) * 1000.0
        )
        self.latency_publisher.publish(planning_latency)
        command = Twist()
        command.linear.x = float(action["linear_velocity"])
        command.angular.z = float(action["angular_velocity"])
        self.publish_command(command)
        self.publish_path(action["trajectory"])
        clearance = Float32()
        clearance.data = float(action["minimum_clearance"])
        self.clearance_publisher.publish(clearance)
        if math.isinf(action["cost"]):
            self.publish_status("BLOCKED")
        elif action.get("recovery_active"):
            self.publish_status("RECOVERY")
        else:
            self.publish_status("PLANNING")

    def destroy_node(self):
        if rclpy.ok():
            self.publish_command(Twist())
            self.publish_path([])
            self.publish_status("STOPPED")
        return super().destroy_node()


def main():
    rclpy.init()
    node = DWAControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

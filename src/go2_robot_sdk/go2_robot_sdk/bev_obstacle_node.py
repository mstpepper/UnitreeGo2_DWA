import math
import time
from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from go2_interfaces.msg import Obstacle2D, Obstacle2DArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

from .dwa_planner import DWAPlanner


class BevObstacleNode(Node):
    """Create a simple bird's-eye obstacle list from RGB, depth, and pose."""

    def __init__(self):
        super().__init__("bev_obstacle_node")
        self.declare_parameter("detection_threshold", 0.5)
        self.declare_parameter("detections_topic", "/detected_objects")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("max_detection_depth_delta", 0.15)
        self.declare_parameter("max_camera_info_delta", 0.5)
        self.declare_parameter("sensor_buffer_duration", 5.0)
        self.declare_parameter("max_detection_age", 2.0)
        self.declare_parameter("obstacles_topic", "/bev/obstacles")
        self.declare_parameter("markers_topic", "/bev/markers")

        self.detection_threshold = float(
            self.get_parameter("detection_threshold").get_parameter_value().double_value
        )
        self.detections_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value
        )
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.max_detection_depth_delta = float(
            self.get_parameter("max_detection_depth_delta")
            .get_parameter_value()
            .double_value
        )
        self.max_camera_info_delta = float(
            self.get_parameter("max_camera_info_delta").value
        )
        self.sensor_buffer_duration = float(
            self.get_parameter("sensor_buffer_duration").value
        )
        self.max_detection_age = float(
            self.get_parameter("max_detection_age").value
        )
        self.obstacles_topic = self.get_parameter("obstacles_topic").value
        self.markers_topic = self.get_parameter("markers_topic").value

        self.bridge = CvBridge()
        self.dwa = DWAPlanner()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_buffer = deque(maxlen=300)
        self.camera_info_buffer = deque(maxlen=300)
        self.obstacles_publisher = self.create_publisher(
            Obstacle2DArray, self.obstacles_topic, 10
        )
        self.markers_publisher = self.create_publisher(
            MarkerArray, self.markers_topic, 10
        )
        self.latency_publisher = self.create_publisher(
            Float32, "/metrics/bev_latency_ms", 10
        )

        self.create_subscription(
            Detection2DArray, self.detections_topic, self.detections_cb, 10
        )
        self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_cb,
            qos_profile_sensor_data,
        )

    def detections_cb(self, msg):
        callback_start = time.perf_counter()
        try:
            self.process_detections(msg)
        finally:
            latency = Float32()
            latency.data = float(
                (time.perf_counter() - callback_start) * 1000.0
            )
            self.latency_publisher.publish(latency)

    def depth_cb(self, msg):
        self.depth_buffer.append(msg)
        self.prune_buffer(self.depth_buffer)

    def camera_info_cb(self, msg):
        self.camera_info_buffer.append(msg)
        self.prune_buffer(self.camera_info_buffer)

    def pixel_to_horizontal_angle(self, u, cam_info):
        fx = cam_info.k[0]
        cx = cam_info.k[2]
        return math.atan2(u - cx, fx)

    def depth_to_local_bev(self, depth_m, theta_x, box_width_px, cam_info):
        angular_width = 2.0 * math.atan2(box_width_px / 2.0, cam_info.k[0])
        estimated_width = 2.0 * depth_m * math.tan(angular_width / 2.0)
        # Camera optical coordinates are x-right, y-down, z-forward.
        camera_x = depth_m * math.sin(theta_x)
        camera_z = depth_m * math.cos(theta_x)
        return camera_x, camera_z, estimated_width

    def camera_point_to_odom(self, camera_x, camera_z, depth_msg, detections_msg):
        source_frame = (
            depth_msg.header.frame_id
            or detections_msg.header.frame_id
        )
        if not source_frame:
            self.get_logger().warning("Camera message has no frame_id; cannot transform obstacle")
            return None

        point = PointStamped()
        point.header = depth_msg.header
        point.header.frame_id = source_frame
        point.point.x = float(camera_x)
        point.point.y = 0.0
        point.point.z = float(camera_z)

        try:
            transform = self.tf_buffer.lookup_transform(
                "odom",
                source_frame,
                Time.from_msg(point.header.stamp),
                timeout=Duration(seconds=0.05),
            )
            return do_transform_point(point, transform)
        except TransformException as exc:
            self.get_logger().warning(
                f"Cannot transform obstacle from {source_frame} to odom: {exc}"
            )
            return None

    @staticmethod
    def stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def prune_buffer(self, message_buffer):
        if not message_buffer:
            return
        newest_time = self.stamp_to_seconds(message_buffer[-1].header.stamp)
        while (
            len(message_buffer) > 1
            and newest_time
            - self.stamp_to_seconds(message_buffer[0].header.stamp)
            > self.sensor_buffer_duration
        ):
            message_buffer.popleft()

    def closest_message(self, message_buffer, target_time):
        if not message_buffer:
            return None
        return min(
            message_buffer,
            key=lambda msg: abs(
                self.stamp_to_seconds(msg.header.stamp) - target_time
            ),
        )

    def process_detections(self, detections_msg):
        detection_time = self.stamp_to_seconds(detections_msg.header.stamp)
        now = self.get_clock().now().nanoseconds * 1e-9
        if (
            detection_time > 0.0
            and self.max_detection_age > 0.0
            and now - detection_time > self.max_detection_age
        ):
            self.get_logger().warning(
                f"Skipping stale detections ({now - detection_time:.3f} seconds old)"
            )
            return

        selected_depth = self.closest_message(self.depth_buffer, detection_time)
        selected_cam_info = self.closest_message(
            self.camera_info_buffer, detection_time
        )
        if selected_depth is None or selected_cam_info is None:
            self.get_logger().warning(
                "Skipping detections: synchronized depth or camera info is unavailable"
            )
            return

        depth_time = self.stamp_to_seconds(selected_depth.header.stamp)
        if abs(detection_time - depth_time) > self.max_detection_depth_delta:
            self.get_logger().warning(
                "Skipping detections: RGB detection and depth timestamps differ by "
                f"{abs(detection_time - depth_time):.3f} seconds"
            )
            return

        camera_info_time = self.stamp_to_seconds(selected_cam_info.header.stamp)
        if (
            camera_info_time > 0.0
            and abs(detection_time - camera_info_time) > self.max_camera_info_delta
        ):
            self.get_logger().warning(
                "Skipping detections: RGB detection and camera-info timestamps "
                f"differ by {abs(detection_time - camera_info_time):.3f} seconds"
            )
            return

        depth_img = self.bridge.imgmsg_to_cv2(selected_depth, desired_encoding="passthrough")
        if depth_img is None:
            return

        depth_img = depth_img.astype(np.float32)
        if selected_depth.encoding.upper() in ("16UC1", "MONO16"):
            depth_img = depth_img / 1000.0
        depth_img = np.where(depth_img > 0.0, depth_img, np.nan)

        height, width = depth_img.shape[:2]
        if (
            selected_cam_info.width not in (0, width)
            or selected_cam_info.height not in (0, height)
        ):
            self.get_logger().warning(
                "Skipping detections: aligned depth and camera-info resolutions differ"
            )
            return

        fx = float(selected_cam_info.k[0])
        cx = float(selected_cam_info.k[2])
        if not math.isfinite(fx) or fx <= 0.0 or not math.isfinite(cx):
            self.get_logger().warning(
                "Skipping detections: camera intrinsics are invalid"
            )
            return

        obstacle_array = Obstacle2DArray()
        obstacle_array.header = detections_msg.header
        obstacle_array.header.frame_id = "odom"

        for detection in detections_msg.detections:
            if not detection.results:
                continue

            hypothesis = detection.results[0].hypothesis
            score = float(hypothesis.score)
            if score < self.detection_threshold:
                continue

            label = hypothesis.class_id
            box_center_u = float(detection.bbox.center.position.x)
            box_center_v = float(detection.bbox.center.position.y)
            box_width_px = float(detection.bbox.size_x)
            theta_x = self.pixel_to_horizontal_angle(box_center_u, selected_cam_info)

            cx = int(max(0, min(width - 1, box_center_u)))
            cy = int(max(0, min(height - 1, box_center_v)))

            depth_val = depth_img[cy, cx]
            if np.isnan(depth_val) or depth_val <= 0.0:
                continue

            camera_x, camera_z, estimated_width = self.depth_to_local_bev(
                float(depth_val), theta_x, float(box_width_px), selected_cam_info
            )

            obstacle_in_odom = self.camera_point_to_odom(
                camera_x, camera_z, selected_depth, detections_msg
            )
            if obstacle_in_odom is None:
                continue

            world_x = obstacle_in_odom.point.x
            world_y = obstacle_in_odom.point.y
            obstacle = Obstacle2D()
            obstacle.x = float(world_x)
            obstacle.y = float(world_y)
            obstacle.radius = max(0.05, float(estimated_width) / 2.0)
            obstacle.confidence = score
            obstacle.label = label
            obstacle_array.obstacles.append(obstacle)

            self.get_logger().info(
                f"detected label={label} score={score:.2f} "
                f"x={world_x:.2f} y={world_y:.2f} width={estimated_width:.2f}"
            )

        self.obstacles_publisher.publish(obstacle_array)
        self.publish_markers(obstacle_array)

    def publish_markers(self, obstacle_array):
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for index, obstacle in enumerate(obstacle_array.obstacles):
            body = Marker()
            body.header = obstacle_array.header
            body.ns = "bev_obstacles"
            body.id = index * 2
            body.type = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = obstacle.x
            body.pose.position.y = obstacle.y
            body.pose.position.z = 0.15
            body.pose.orientation.w = 1.0
            body.scale.x = obstacle.radius * 2.0
            body.scale.y = obstacle.radius * 2.0
            body.scale.z = 0.3
            body.color.r = 1.0
            body.color.g = 0.25
            body.color.b = 0.0
            body.color.a = 0.75
            marker_array.markers.append(body)

            text = Marker()
            text.header = obstacle_array.header
            text.ns = "bev_labels"
            text.id = index * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = obstacle.x
            text.pose.position.y = obstacle.y
            text.pose.position.z = 0.55
            text.pose.orientation.w = 1.0
            text.scale.z = 0.18
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = (
                f"{obstacle.label} {obstacle.confidence:.2f}\n"
                f"({obstacle.x:.2f}, {obstacle.y:.2f})"
            )
            marker_array.markers.append(text)

        self.markers_publisher.publish(marker_array)


def main():
    rclpy.init()
    node = BevObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

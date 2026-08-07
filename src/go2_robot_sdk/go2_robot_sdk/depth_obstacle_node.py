import math
import time
from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from go2_interfaces.msg import Obstacle2D, Obstacle2DArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def transform_points(points, transform):
    """Apply a geometry_msgs Transform to an (N, 3) NumPy array."""
    q = transform.rotation
    q_values = np.asarray([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = float(np.linalg.norm(q_values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("TF rotation is invalid")
    x, y, z, w = q_values / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    translation = np.asarray(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=np.float64,
    )
    return points @ rotation.T + translation


def cluster_occupied_cells(points_xy, cell_size):
    """Return 8-connected components of occupied ground-plane grid cells."""
    cell_indices = np.floor(points_xy / cell_size).astype(np.int64)
    cells = {}
    for point_index, cell in enumerate(cell_indices):
        key = (int(cell[0]), int(cell[1]))
        cells.setdefault(key, []).append(point_index)

    clusters = []
    unseen = set(cells)
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        indices = []
        while queue:
            cell = queue.popleft()
            indices.extend(cells[cell])
            cx, cy = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
        clusters.append(np.asarray(indices, dtype=np.int64))
    return clusters


class DepthObstacleNode(Node):
    """Extract unlabeled 2-D obstacles directly from an aligned depth image."""

    def __init__(self):
        super().__init__("depth_obstacle_node")
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter("obstacles_topic", "/bev/obstacles")
        self.declare_parameter("markers_topic", "/bev/markers")
        self.declare_parameter("base_frame", "robot0/base_link")
        self.declare_parameter("output_frame", "odom")
        self.declare_parameter("pixel_stride", 4)
        self.declare_parameter("minimum_depth", 0.25)
        self.declare_parameter("maximum_depth", 4.0)
        self.declare_parameter("minimum_height", 0.06)
        self.declare_parameter("maximum_height", 1.50)
        self.declare_parameter("grid_cell_size", 0.10)
        self.declare_parameter("minimum_cluster_points", 8)
        self.declare_parameter("minimum_radius", 0.08)
        self.declare_parameter("maximum_radius", 1.00)
        self.declare_parameter("obstacle_inflation", 0.05)
        self.declare_parameter("max_camera_info_age", 2.0)

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        self.obstacles_topic = str(self.get_parameter("obstacles_topic").value)
        self.markers_topic = str(self.get_parameter("markers_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.pixel_stride = max(1, int(self.get_parameter("pixel_stride").value))
        self.minimum_depth = float(self.get_parameter("minimum_depth").value)
        self.maximum_depth = float(self.get_parameter("maximum_depth").value)
        self.minimum_height = float(self.get_parameter("minimum_height").value)
        self.maximum_height = float(self.get_parameter("maximum_height").value)
        self.grid_cell_size = float(self.get_parameter("grid_cell_size").value)
        self.minimum_cluster_points = max(
            1, int(self.get_parameter("minimum_cluster_points").value)
        )
        self.minimum_radius = float(self.get_parameter("minimum_radius").value)
        self.maximum_radius = float(self.get_parameter("maximum_radius").value)
        self.obstacle_inflation = float(
            self.get_parameter("obstacle_inflation").value
        )
        self.max_camera_info_age = float(
            self.get_parameter("max_camera_info_age").value
        )

        if self.minimum_depth < 0.0 or self.maximum_depth <= self.minimum_depth:
            raise ValueError("Depth limits are invalid")
        if self.maximum_height <= self.minimum_height:
            raise ValueError("Height limits are invalid")
        if self.grid_cell_size <= 0.0:
            raise ValueError("grid_cell_size must be positive")

        self.bridge = CvBridge()
        self.camera_info = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.obstacles_publisher = self.create_publisher(
            Obstacle2DArray, self.obstacles_topic, 10
        )
        self.markers_publisher = self.create_publisher(
            MarkerArray, self.markers_topic, 10
        )
        self.latency_publisher = self.create_publisher(
            Float32, "/metrics/depth_bev_latency_ms", 10
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data
        )
        self.get_logger().info(
            "Non-AI depth obstacle extraction enabled; "
            f"publishing {self.obstacles_topic} in {self.output_frame}"
        )

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def camera_info_cb(self, message):
        self.camera_info = message

    def depth_cb(self, message):
        callback_start = time.perf_counter()
        try:
            self.process_depth(message)
        finally:
            latency = Float32()
            latency.data = float((time.perf_counter() - callback_start) * 1000.0)
            self.latency_publisher.publish(latency)

    def publish_empty(self, depth_message):
        result = Obstacle2DArray()
        result.header.stamp = depth_message.header.stamp
        result.header.frame_id = self.output_frame
        self.obstacles_publisher.publish(result)
        self.publish_markers(result)

    def process_depth(self, depth_message):
        camera_info = self.camera_info
        if camera_info is None:
            self.get_logger().warning("Waiting for camera_info", throttle_duration_sec=2.0)
            return

        depth_time = self.stamp_seconds(depth_message.header.stamp)
        info_time = self.stamp_seconds(camera_info.header.stamp)
        if (
            depth_time > 0.0
            and info_time > 0.0
            and abs(depth_time - info_time) > self.max_camera_info_age
        ):
            self.get_logger().warning(
                "Depth and camera_info timestamps are too far apart",
                throttle_duration_sec=2.0,
            )
            return

        source_frame = depth_message.header.frame_id or camera_info.header.frame_id
        if not source_frame:
            self.get_logger().warning("Depth image has no optical frame_id")
            return

        depth = self.bridge.imgmsg_to_cv2(
            depth_message, desired_encoding="passthrough"
        )
        if depth is None or depth.ndim != 2:
            return
        depth = depth.astype(np.float32)
        if depth_message.encoding.upper() in ("16UC1", "MONO16"):
            depth *= 0.001

        height, width = depth.shape
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0 or not all(
            math.isfinite(value) for value in (fx, fy, cx, cy)
        ):
            self.get_logger().warning("Camera intrinsics are invalid")
            return
        if camera_info.width not in (0, width) or camera_info.height not in (
            0,
            height,
        ):
            self.get_logger().warning("Depth and camera_info resolutions differ")
            return

        stamp = Time.from_msg(depth_message.header.stamp)
        try:
            camera_to_base = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=0.10),
            )
            camera_to_output = self.tf_buffer.lookup_transform(
                self.output_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=0.10),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"Cannot transform depth points from {source_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        rows = np.arange(0, height, self.pixel_stride, dtype=np.float32)
        columns = np.arange(0, width, self.pixel_stride, dtype=np.float32)
        uu, vv = np.meshgrid(columns, rows)
        sampled_depth = depth[:: self.pixel_stride, :: self.pixel_stride]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= self.minimum_depth)
            & (sampled_depth <= self.maximum_depth)
        )
        if not np.any(valid):
            self.publish_empty(depth_message)
            return

        z_forward = sampled_depth[valid]
        x_right = (uu[valid] - cx) * z_forward / fx
        y_down = (vv[valid] - cy) * z_forward / fy
        camera_points = np.column_stack((x_right, y_down, z_forward))

        try:
            base_points = transform_points(
                camera_points, camera_to_base.transform
            )
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        obstacle_mask = (
            (base_points[:, 2] >= self.minimum_height)
            & (base_points[:, 2] <= self.maximum_height)
            & (base_points[:, 0] >= 0.0)
        )
        camera_points = camera_points[obstacle_mask]
        base_points = base_points[obstacle_mask]
        if len(base_points) == 0:
            self.publish_empty(depth_message)
            return

        result = Obstacle2DArray()
        result.header.stamp = depth_message.header.stamp
        result.header.frame_id = self.output_frame
        clusters = cluster_occupied_cells(
            base_points[:, :2], self.grid_cell_size
        )
        for cluster_indices in clusters:
            if len(cluster_indices) < self.minimum_cluster_points:
                continue
            cluster_base = base_points[cluster_indices]
            cluster_camera = camera_points[cluster_indices]
            center_base = np.median(cluster_base, axis=0)
            radial_extent = np.linalg.norm(
                cluster_base[:, :2] - center_base[:2], axis=1
            )
            radius = float(np.percentile(radial_extent, 90.0))
            radius = max(self.minimum_radius, radius + self.obstacle_inflation)
            radius = min(self.maximum_radius, radius)

            center_camera = np.median(cluster_camera, axis=0, keepdims=True)
            center_output = transform_points(
                center_camera, camera_to_output.transform
            )[0]
            if not np.all(np.isfinite(center_output[:2])):
                continue

            obstacle = Obstacle2D()
            obstacle.x = float(center_output[0])
            obstacle.y = float(center_output[1])
            obstacle.radius = radius
            obstacle.confidence = min(
                1.0, float(len(cluster_indices)) / (4.0 * self.minimum_cluster_points)
            )
            obstacle.label = "depth_obstacle"
            result.obstacles.append(obstacle)

        self.obstacles_publisher.publish(result)
        self.publish_markers(result)

    def publish_markers(self, obstacles):
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)
        for index, obstacle in enumerate(obstacles.obstacles):
            marker = Marker()
            marker.header = obstacles.header
            marker.ns = "depth_obstacles"
            marker.id = index
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle.x
            marker.pose.position.y = obstacle.y
            marker.pose.position.z = 0.15
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2.0 * obstacle.radius
            marker.scale.y = 2.0 * obstacle.radius
            marker.scale.z = 0.30
            marker.color.r = 0.10
            marker.color.g = 0.55
            marker.color.b = 1.00
            marker.color.a = 0.75
            marker_array.markers.append(marker)
        self.markers_publisher.publish(marker_array)


def main():
    rclpy.init()
    node = DepthObstacleNode()
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

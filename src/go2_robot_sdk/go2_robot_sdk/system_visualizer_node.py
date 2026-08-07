"""Headless, read-only Matplotlib dashboard for the Go2 navigation pipeline."""

import io
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist  # noqa: E402
from go2_interfaces.msg import Obstacle2DArray  # noqa: E402
from nav_msgs.msg import Odometry, Path  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Float32, String  # noqa: E402


PAGE = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Go2 Navigation Visualizer</title>
  <style>
    html, body { margin: 0; background: #101419; color: #e8edf2;
                 font-family: sans-serif; text-align: center; }
    h1 { margin: 12px 0 4px; font-size: 20px; font-weight: 600; }
    p { margin: 0 0 10px; color: #aeb9c4; font-size: 13px; }
    img { width: min(98vw, 1500px); height: auto; background: white;
          border: 1px solid #34404b; }
  </style>
</head>
<body>
  <h1>Go2 Navigation Visualizer</h1>
  <p>Read-only dashboard. It cannot publish robot commands.</p>
  <img id="plot" alt="Waiting for the first rendered frame">
  <script>
    const image = document.getElementById("plot");
    function refresh() {
      image.src = "/plot.png?t=" + Date.now();
    }
    image.onload = () => setTimeout(refresh, 250);
    image.onerror = () => setTimeout(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


class DashboardServer(ThreadingHTTPServer):
    """HTTP server carrying a reference to the visualizer node."""

    daemon_threads = True

    def __init__(self, address, handler, visualizer):
        super().__init__(address, handler)
        self.visualizer = visualizer


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard and the latest cached Matplotlib frame."""

    def do_GET(self):  # pylint: disable=invalid-name
        if self.path == "/" or self.path.startswith("/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE)
            return
        if self.path.startswith("/plot.png"):
            image = self.server.visualizer.get_latest_png()
            if image is None:
                self._send(503, "text/plain; charset=utf-8", b"Plot not ready")
            else:
                self._send(200, "image/png", image, no_cache=True)
            return
        if self.path.startswith("/status.json"):
            payload = json.dumps(self.server.visualizer.status()).encode("utf-8")
            self._send(200, "application/json", payload, no_cache=True)
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _send(self, status, content_type, payload, no_cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if no_cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        return


class SystemVisualizerNode(Node):
    """Subscribe to perception/planning/velocity topics and render a dashboard."""

    def __init__(self):
        super().__init__("go2_system_visualizer")
        self.declare_parameter("bind_address", "127.0.0.1")
        self.declare_parameter("port", 8080)
        self.declare_parameter("render_rate", 2.0)
        self.declare_parameter("stale_timeout", 2.0)
        self.declare_parameter("view_forward", 4.0)
        self.declare_parameter("view_side", 3.0)
        self.declare_parameter("history_seconds", 20.0)
        self.declare_parameter("robot_length", 0.7)
        self.declare_parameter("robot_width", 0.35)
        self.declare_parameter("bev_topic", "/bev/obstacles")
        self.declare_parameter("odom_topic", "/robot0/odom")
        self.declare_parameter("goal_topic", "/dwa/goal")
        self.declare_parameter("command_topic", "/dwa/cmd_vel_preview")
        self.declare_parameter("dwa_path_topic", "/dwa/predicted_path")
        self.declare_parameter("dwa_status_topic", "/dwa/status")
        self.declare_parameter(
            "dwa_clearance_topic", "/dwa/minimum_clearance"
        )
        self.declare_parameter("preview_only", True)
        self.declare_parameter("safe_command_topic", "/cmd_vel_safe_debug")
        self.declare_parameter("output_topic", "/robot0/cmd_vel_out")

        self.bind_address = str(self.get_parameter("bind_address").value)
        self.port = int(self.get_parameter("port").value)
        self.render_rate = max(0.2, float(self.get_parameter("render_rate").value))
        self.stale_timeout = max(0.1, float(self.get_parameter("stale_timeout").value))
        self.view_forward = max(0.5, float(self.get_parameter("view_forward").value))
        self.view_side = max(0.5, float(self.get_parameter("view_side").value))
        self.history_seconds = max(2.0, float(self.get_parameter("history_seconds").value))
        self.robot_length = max(0.05, float(self.get_parameter("robot_length").value))
        self.robot_width = max(0.05, float(self.get_parameter("robot_width").value))
        self.preview_only = bool(self.get_parameter("preview_only").value)

        self.data_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_png = None
        self.obstacles = []
        self.bev_frame = ""
        self.goal = None
        self.robot_pose = None
        self.dwa_path = []
        self.dwa_status = "NOT CONNECTED"
        self.dwa_clearance = None
        self.last_update = {}
        self.histories = {
            "command": deque(),
            "safe": deque(),
            "output": deque(),
        }

        self.create_subscription(
            Obstacle2DArray,
            str(self.get_parameter("bev_topic").value),
            self.bev_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_cb,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("goal_topic").value),
            self.goal_cb,
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("command_topic").value),
            lambda msg: self.twist_cb("command", msg),
            10,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("dwa_path_topic").value),
            self.dwa_path_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("dwa_status_topic").value),
            self.dwa_status_cb,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("dwa_clearance_topic").value),
            self.dwa_clearance_cb,
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("safe_command_topic").value),
            lambda msg: self.twist_cb("safe", msg),
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("output_topic").value),
            lambda msg: self.twist_cb("output", msg),
            10,
        )

        self.http_server = DashboardServer(
            (self.bind_address, self.port), DashboardHandler, self
        )
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            name="visualizer-http",
            daemon=True,
        )
        self.render_thread = threading.Thread(
            target=self.render_loop,
            name="visualizer-render",
            daemon=True,
        )
        self.http_thread.start()
        self.render_thread.start()
        self.get_logger().info(
            f"Read-only dashboard listening on http://{self.bind_address}:{self.port}"
        )

    def bev_cb(self, msg):
        now = time.monotonic()
        with self.data_lock:
            self.obstacles = [
                (item.x, item.y, item.radius, item.confidence, item.label)
                for item in msg.obstacles
            ]
            self.bev_frame = msg.header.frame_id
            self.last_update["bev"] = now

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        position = msg.pose.pose.position
        with self.data_lock:
            self.robot_pose = (position.x, position.y, yaw)
            self.last_update["odom"] = time.monotonic()

    def goal_cb(self, msg):
        with self.data_lock:
            self.goal = (msg.pose.position.x, msg.pose.position.y)
            self.last_update["goal"] = time.monotonic()

    def twist_cb(self, name, msg):
        now = time.monotonic()
        sample = (now, msg.linear.x, msg.linear.y, msg.angular.z)
        with self.data_lock:
            history = self.histories[name]
            history.append(sample)
            cutoff = now - self.history_seconds
            while history and history[0][0] < cutoff:
                history.popleft()
            self.last_update[name] = now

    def dwa_path_cb(self, msg):
        with self.data_lock:
            self.dwa_path = [
                (pose.pose.position.x, pose.pose.position.y)
                for pose in msg.poses
            ]
            self.last_update["dwa_path"] = time.monotonic()

    def dwa_status_cb(self, msg):
        with self.data_lock:
            self.dwa_status = msg.data
            self.last_update["dwa_status"] = time.monotonic()

    def dwa_clearance_cb(self, msg):
        with self.data_lock:
            self.dwa_clearance = float(msg.data)
            self.last_update["dwa_clearance"] = time.monotonic()

    def snapshot(self):
        with self.data_lock:
            return {
                "obstacles": list(self.obstacles),
                "bev_frame": self.bev_frame,
                "goal": self.goal,
                "robot_pose": self.robot_pose,
                "dwa_path": list(self.dwa_path),
                "dwa_status": self.dwa_status,
                "dwa_clearance": self.dwa_clearance,
                "last_update": dict(self.last_update),
                "histories": {
                    name: list(history) for name, history in self.histories.items()
                },
            }

    def age_text(self, snapshot, name, now):
        updated = snapshot["last_update"].get(name)
        if updated is None:
            return "NOT CONNECTED", "#a82a2a"
        age = now - updated
        if age > self.stale_timeout:
            return f"STALE ({age:.1f}s)", "#d97706"
        return f"LIVE ({age:.2f}s)", "#16803b"

    @staticmethod
    def rotated_rectangle(x, y, yaw, length, width):
        corners = [
            (length / 2.0, width / 2.0),
            (length / 2.0, -width / 2.0),
            (-length / 2.0, -width / 2.0),
            (-length / 2.0, width / 2.0),
        ]
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return [
            (x + cosine * px - sine * py, y + sine * px + cosine * py)
            for px, py in corners
        ]

    def render_loop(self):
        period = 1.0 / self.render_rate
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                image = self.render_png(self.snapshot())
                with self.image_lock:
                    self.latest_png = image
            except Exception as exc:  # keep dashboard alive on a bad sample
                self.get_logger().error(f"Visualizer render failed: {exc}")
            remaining = max(0.0, period - (time.monotonic() - started))
            self.stop_event.wait(remaining)

    def render_png(self, snapshot):
        now = time.monotonic()
        figure = plt.figure(figsize=(13, 7.2), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, width_ratios=(1.3, 1.0))
        bev_axis = figure.add_subplot(grid[:, 0])
        linear_axis = figure.add_subplot(grid[0, 1])
        yaw_axis = figure.add_subplot(grid[1, 1])

        robot_x, robot_y, robot_yaw = snapshot["robot_pose"] or (0.0, 0.0, 0.0)
        robot_shape = self.rotated_rectangle(
            robot_x,
            robot_y,
            robot_yaw,
            self.robot_length,
            self.robot_width,
        )
        bev_axis.add_patch(
            Polygon(robot_shape, closed=True, facecolor="#2563eb", alpha=0.75)
        )
        bev_axis.arrow(
            robot_x,
            robot_y,
            0.5 * math.cos(robot_yaw),
            0.5 * math.sin(robot_yaw),
            width=0.025,
            color="#173f91",
            length_includes_head=True,
        )

        for x_pos, y_pos, radius, confidence, label in snapshot["obstacles"]:
            radius = max(0.03, radius)
            bev_axis.add_patch(
                Circle((x_pos, y_pos), radius, color="#dc2626", alpha=0.35)
            )
            bev_axis.plot(x_pos, y_pos, "o", color="#991b1b", markersize=4)
            bev_axis.text(
                x_pos,
                y_pos + radius + 0.05,
                f"{label} {confidence:.0%}\n({x_pos:.2f}, {y_pos:.2f}) m",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if snapshot["goal"] is not None:
            goal_x, goal_y = snapshot["goal"]
            bev_axis.plot(goal_x, goal_y, marker="*", color="#16a34a", markersize=16)
            bev_axis.text(goal_x, goal_y + 0.12, "DWA goal", ha="center", fontsize=9)

        if snapshot["dwa_path"]:
            bev_axis.plot(
                [point[0] for point in snapshot["dwa_path"]],
                [point[1] for point in snapshot["dwa_path"]],
                color="#f97316",
                linewidth=2.0,
                label="DWA predicted path",
            )
            bev_axis.legend(loc="upper right", fontsize=8)

        command_history = snapshot["histories"]["command"]
        if command_history:
            _, linear_x, linear_y, _ = command_history[-1]
            bev_axis.arrow(
                robot_x,
                robot_y,
                linear_x,
                linear_y,
                width=0.02,
                color="#9333ea",
                length_includes_head=True,
            )

        bev_status, bev_color = self.age_text(snapshot, "bev", now)
        bev_axis.set_title(
            f"Bird's-eye view — {bev_status} — frame: "
            f"{snapshot['bev_frame'] or 'unknown'}",
            color=bev_color,
            fontweight="bold",
        )
        bev_axis.set_xlim(-1.0, self.view_forward)
        bev_axis.set_ylim(-self.view_side, self.view_side)
        bev_axis.set_aspect("equal", adjustable="box")
        bev_axis.set_xlabel("x (m): forward →")
        bev_axis.set_ylabel("y (m): left + / right −")
        bev_axis.grid(True, alpha=0.3)

        styles = {
            "command": ("requested", "#9333ea"),
            "safe": ("safety-gate", "#ea580c"),
            "output": ("robot output", "#16a34a"),
        }
        for name, history in snapshot["histories"].items():
            if not history:
                continue
            label, color = styles[name]
            times = [sample[0] - now for sample in history]
            linear_axis.plot(
                times, [sample[1] for sample in history],
                color=color, label=f"{label} vx"
            )
            linear_axis.plot(
                times, [sample[2] for sample in history],
                color=color, linestyle="--", label=f"{label} vy"
            )
            yaw_axis.plot(
                times, [sample[3] for sample in history],
                color=color, label=label
            )

        linear_axis.axhline(0.0, color="black", linewidth=0.6)
        linear_axis.set_title("Velocity publisher / safety gate")
        linear_axis.set_ylabel("linear velocity (m/s)")
        linear_axis.set_xlim(-self.history_seconds, 0.0)
        linear_axis.grid(True, alpha=0.3)
        if linear_axis.lines[1:]:
            linear_axis.legend(fontsize=8, loc="upper left")

        yaw_axis.axhline(0.0, color="black", linewidth=0.6)
        yaw_axis.set_xlabel("seconds before now")
        yaw_axis.set_ylabel("yaw rate (rad/s)")
        yaw_axis.set_xlim(-self.history_seconds, 0.0)
        yaw_axis.grid(True, alpha=0.3)
        if yaw_axis.lines[1:]:
            yaw_axis.legend(fontsize=8, loc="upper left")

        connected = []
        for name in ("odom", "goal", "command", "safe", "output"):
            text, _ = self.age_text(snapshot, name, now)
            connected.append(f"{name}: {text}")
        clearance = snapshot["dwa_clearance"]
        clearance_text = (
            "unknown"
            if clearance is None
            else ("infinite" if math.isinf(clearance) else f"{clearance:.2f} m")
        )
        safety_banner = (
            "PREVIEW ONLY - MOTION COMMAND PUBLISHING DISABLED"
            if self.preview_only
            else "MOTION-CAPABLE MODE - VERIFY SAFETY GATE"
        )
        figure.suptitle(
            f"{safety_banner}\n"
            f"DWA: {snapshot['dwa_status']} | clearance: {clearance_text}\n"
            + "  |  ".join(connected),
            fontsize=10,
            color="#b91c1c" if not self.preview_only else "#166534",
            fontweight="bold",
        )
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110)
        plt.close(figure)
        return buffer.getvalue()

    def get_latest_png(self):
        with self.image_lock:
            return self.latest_png

    def status(self):
        snapshot = self.snapshot()
        now = time.monotonic()
        return {
            name: self.age_text(snapshot, name, now)[0]
            for name in (
                "bev",
                "odom",
                "goal",
                "dwa_path",
                "dwa_status",
                "command",
                "safe",
                "output",
            )
        }

    def destroy_node(self):
        self.stop_event.set()
        self.http_server.shutdown()
        self.http_server.server_close()
        self.render_thread.join(timeout=2.0)
        self.http_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SystemVisualizerNode()
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
